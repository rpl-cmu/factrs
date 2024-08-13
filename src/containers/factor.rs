use crate::{
    containers::{Key, Values},
    dtype,
    linalg::{AllocatorBuffer, Const, DefaultAllocator, DiffResult, DualAllocator, MatrixBlock},
    linear::LinearFactor,
    noise::{NoiseModel, NoiseModelSafe, UnitNoise},
    residuals::{Residual, ResidualSafe},
    robust::{RobustCostSafe, L2},
};

/// Main structure to represent a factor in the graph.
///
/// $$ \blue{\rho_i}(||\purple{r_i}(\green{\Theta})||_{\red{\Sigma_i}} ) $$
///
/// Factors are the main building block of the factor graph. They are composed
/// of four pieces:
/// - <green>Keys</green>: The variables that the factor depends on, given by a
///   slice of [Keys](Key).
/// - <purple>Residual</purple>: The vector-valued function that computes the
///   error of the factor given a set of values, from the
///   [residual](crate::residuals) module.
/// - <red>Noise Model</red>: The noise model describes the uncertainty of the
///   residual, given by the traits in the [noise](crate::noise) module.
/// - <blue>Robust Kernel</blue>: The robust kernel weights the error of the
///   factor, given by the traits in the [robust](crate::robust) module.
///
/// Constructors are available for a number of default cases including default
/// robust kernel [L2], default noise model [UnitNoise]. Keys and residual are
/// always required.
///
/// During optimization the factor is linearized around a set of values into a
/// [LinearFactor].
///
///  ```
/// # use factrs::prelude::*;
/// let prior = VectorVar3::new(1.0, 2.0, 3.0);
/// let residual = PriorResidual::new(prior);
/// let noise = GaussianNoise::<3>::from_diag_sigmas(1e-1, 2e-1, 3e-1);
/// let robust = GemanMcClure::default();
/// let factor = Factor::new_full(&[X(0)], residual, noise, robust);
/// ```
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Factor {
    keys: Vec<Key>,
    residual: Box<dyn ResidualSafe>,
    noise: Box<dyn NoiseModelSafe>,
    robust: Box<dyn RobustCostSafe>,
}

impl Factor {
    /// Build a new factor from a set of keys and a residual.
    ///
    /// Keys will be compile-time checked to ensure the size is consistent with
    /// the residual. Noise will be set to [UnitNoise] and robust kernel to
    /// [L2].
    pub fn new_base<const NUM_VARS: usize, const DIM_OUT: usize, R>(
        keys: &[Key; NUM_VARS],
        residual: R,
    ) -> Self
    where
        R: 'static + Residual<NumVars = Const<NUM_VARS>, DimOut = Const<DIM_OUT>> + ResidualSafe,
        AllocatorBuffer<R::DimIn>: Sync + Send,
        DefaultAllocator: DualAllocator<R::DimIn>,
        UnitNoise<DIM_OUT>: NoiseModelSafe,
    {
        Self {
            keys: keys.to_vec(),
            residual: Box::new(residual),
            noise: Box::new(UnitNoise::<DIM_OUT>),
            robust: Box::new(L2),
        }
    }

    /// Build a new factor from a set of keys, a residual, and a noise model.
    ///
    /// Keys and noise will be compile-time checked to ensure the size is
    /// consistent with the residual. Robust kernel will be set to [L2].
    pub fn new_noise<const NUM_VARS: usize, const DIM_OUT: usize, R, N>(
        keys: &[Key; NUM_VARS],
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

    /// Build a new factor from a set of keys, a residual, a noise model, and a
    /// robust kernel.
    ///
    /// Keys and noise will be compile-time checked to ensure the size is
    /// consistent with the residual.
    pub fn new_full<const NUM_VARS: usize, const DIM_OUT: usize, R, N, C>(
        keys: &[Key; NUM_VARS],
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

    /// Compute the error of the factor given a set of values.
    pub fn error(&self, values: &Values) -> dtype {
        let r = self.residual.residual(values, &self.keys);
        let r = self.noise.whiten_vec(r);
        let norm2 = r.norm_squared();
        self.robust.loss(norm2)
    }

    /// Compute the dimension of the output of the factor.
    pub fn dim_out(&self) -> usize {
        self.residual.dim_out()
    }

    /// Linearize the factor given a set of values into a [LinearFactor].
    pub fn linearize(&self, values: &Values) -> LinearFactor {
        // Compute residual and jacobian
        let DiffResult { value: r, diff: a } = self.residual.residual_jacobian(values, &self.keys);

        // Whiten residual and jacobian
        let r = self.noise.whiten_vec(r);
        let a = self.noise.whiten_mat(a);

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
                *sum += values.get_raw(*k).unwrap().dim();
                out
            })
            .collect::<Vec<_>>();
        let a = MatrixBlock::new(a, idx);

        LinearFactor::new(self.keys.clone(), a, b)
    }

    /// Get the keys of the factor.
    pub fn keys(&self) -> &[Key] {
        &self.keys
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

        let factor = Factor::new_full(&[X(0).into()], residual, noise, robust);

        let f = |x: VectorVar3| {
            let mut values = Values::new();
            values.insert_unchecked(X(0), x);
            factor.error(&values)
        };

        let mut values = Values::new();
        values.insert_unchecked(X(0), x.clone());

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

        let factor = Factor::new_full(&[X(0).into(), X(1).into()], residual, noise, robust);

        let mut values = Values::new();
        values.insert_unchecked(X(0), x.clone());
        values.insert_unchecked(X(1), x);

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
