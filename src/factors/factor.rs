use crate::{
    containers::{Symbol, Values},
    dtype,
    linalg::{AllocatorBuffer, Const, DefaultAllocator, DiffResult, DualAllocator, MatrixBlock},
    linear::LinearFactor,
    noise::{GaussianNoise, NoiseModel, NoiseModelSafe},
    residuals::{Residual, ResidualSafe},
    robust::{RobustCostSafe, L2},
};

#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Factor {
    pub keys: Vec<Symbol>,
    residual: Box<dyn ResidualSafe>,
    noise: Box<dyn NoiseModelSafe>,
    robust: Box<dyn RobustCostSafe>,
}

impl Factor {
    pub fn new_base<const NUM_VARS: usize, const DIM_OUT: usize, R>(
        keys: &[Symbol; NUM_VARS],
        residual: R,
    ) -> Self
    where
        R: 'static + Residual<NumVars = Const<NUM_VARS>, DimOut = Const<DIM_OUT>> + ResidualSafe,
        AllocatorBuffer<R::DimIn>: Sync + Send,
        DefaultAllocator: DualAllocator<R::DimIn>,
        GaussianNoise<DIM_OUT>: NoiseModelSafe,
    {
        Self {
            keys: keys.to_vec(),
            residual: Box::new(residual),
            noise: Box::new(GaussianNoise::<DIM_OUT>::identity()),
            robust: Box::new(L2),
        }
    }

    pub fn new_noise<const NUM_VARS: usize, const DIM_OUT: usize, R, N>(
        keys: &[Symbol; NUM_VARS],
        residual: R,
        noise: N,
    ) -> Self
    where
        R: 'static + Residual<NumVars = Const<NUM_VARS>, DimOut = Const<DIM_OUT>> + ResidualSafe,
        N: 'static + NoiseModel<Dim = Const<DIM_OUT>> + NoiseModelSafe,
        AllocatorBuffer<R::DimIn>: Sync + Send,
        DefaultAllocator: DualAllocator<R::DimIn>,
    {
        Self {
            keys: keys.to_vec(),
            residual: Box::new(residual),
            noise: Box::new(noise),
            robust: Box::new(L2),
        }
    }

    pub fn new_full<const NUM_VARS: usize, const DIM_OUT: usize, R, N, C>(
        keys: &[Symbol; NUM_VARS],
        residual: R,
        noise: N,
        robust: C,
    ) -> Self
    where
        R: 'static + Residual<NumVars = Const<NUM_VARS>, DimOut = Const<DIM_OUT>> + ResidualSafe,
        AllocatorBuffer<R::DimIn>: Sync + Send,
        DefaultAllocator: DualAllocator<R::DimIn>,
        N: 'static + NoiseModel<Dim = Const<DIM_OUT>> + NoiseModelSafe,
        C: 'static + RobustCostSafe,
    {
        Self {
            keys: keys.to_vec(),
            residual: Box::new(residual),
            noise: Box::new(noise),
            robust: Box::new(robust),
        }
    }

    pub fn error(&self, values: &Values) -> dtype {
        let r = self.residual.residual(values, &self.keys);
        let r = self.noise.whiten_vec(r.as_view());
        let norm2 = r.norm_squared();
        self.robust.loss(norm2)
    }

    pub fn dim_out(&self) -> usize {
        self.residual.dim_out()
    }

    pub fn linearize(&self, values: &Values) -> LinearFactor {
        // Compute residual and jacobian
        let DiffResult { value: r, diff: a } = self.residual.residual_jacobian(values, &self.keys);

        // Whiten residual and jacobian
        let r = self.noise.whiten_vec(r.as_view());
        let a = self.noise.whiten_mat(a.as_view());

        // Weight according to robust cost
        let norm2 = r.norm_squared();
        let weight = self.robust.weight(norm2).sqrt();
        let a = weight * a;
        let b = -weight * r;

        // Turn A into a MatrixBlock
        let idx = self
            .keys
            .iter()
            .scan(0, |sum, k| {
                let out = Some(*sum);
                *sum += values.get(k).unwrap().dim();
                out
            })
            .collect::<Vec<_>>();
        let a = MatrixBlock::new(a, idx);

        LinearFactor::new(self.keys.clone(), a, b)
    }
}

#[cfg(test)]
mod tests {

    use matrixcompare::assert_matrix_eq;

    use super::*;
    use crate::{
        containers::X,
        linalg::{Diff, NumericalDiff},
        noise::GaussianNoise,
        residuals::{BetweenResidual, PriorResidual},
        robust::GemanMcClure,
        variables::{Variable, VectorVar3},
    };

    #[cfg(not(feature = "f32"))]
    const PWR: i32 = 6;
    #[cfg(not(feature = "f32"))]
    const TOL: f64 = 1e-6;

    #[cfg(feature = "f32")]
    const PWR: i32 = 3;
    #[cfg(feature = "f32")]
    const TOL: f32 = 1e-3;

    #[test]
    fn linearize_a() {
        let prior = VectorVar3::new(1.0, 2.0, 3.0);
        let x = VectorVar3::identity();

        let residual = PriorResidual::new(prior);
        let noise = GaussianNoise::<3>::from_diag_sigmas(1e-1, 2e-1, 3e-1);
        let robust = GemanMcClure::default();

        let factor = Factor::new_full(&[X(0)], residual, noise, robust);

        let f = |x: VectorVar3| {
            let mut values = Values::new();
            values.insert(X(0), x);
            factor.error(&values)
        };

        let mut values = Values::new();
        values.insert(X(0), x.clone());

        let linear = factor.linearize(&values);
        let grad_got = -linear.a.mat().transpose() * linear.b;
        println!("Received {:}", grad_got);

        let grad_num = NumericalDiff::<PWR>::gradient_1(f, &x).diff;
        println!("Expected {:}", grad_num);

        assert_matrix_eq!(grad_got, grad_num, comp = abs, tol = TOL);
    }

    #[test]
    fn linearize_block() {
        let bet = VectorVar3::new(1.0, 2.0, 3.0);
        let x = <VectorVar3 as Variable>::identity();

        let residual = BetweenResidual::new(bet);
        let noise = GaussianNoise::<3>::from_diag_sigmas(1e-1, 2e-1, 3e-1);
        let robust = GemanMcClure::default();

        let factor = Factor::new_full(&[X(0), X(1)], residual, noise, robust);

        let mut values = Values::new();
        values.insert(X(0), x.clone());
        values.insert(X(1), x);

        let linear = factor.linearize(&values);

        println!("Full Mat {:}", linear.a.mat());
        println!("First Block {:}", linear.a.get_block(0));
        println!("Second Block {:}", linear.a.get_block(1));

        assert_matrix_eq!(
            linear.a.get_block(0),
            linear.a.mat().columns(0, 3),
            comp = float
        );
        assert_matrix_eq!(
            linear.a.get_block(1),
            linear.a.mat().columns(3, 3),
            comp = float
        );
    }
}
