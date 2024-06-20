use crate::containers::{Key, Values};
use crate::dtype;
use crate::linalg::{DiffResult, MatrixBlock};
use crate::linear::LinearFactor;
use crate::noise::{GaussianNoise, NoiseModel};
use crate::residuals::Residual;
use crate::robust::{RobustCost, L2};
use crate::variables::Variable;

pub struct FactorGeneric<K: Key, V: Variable, R: Residual<V>, N: NoiseModel, C: RobustCost> {
    keys: Vec<K>,
    residual: R,
    noise: N,
    robust: C,
    _phantom: std::marker::PhantomData<V>,
}

pub struct FactorFactory<K: Key, V: Variable, R: Residual<V>, N: NoiseModel, C: RobustCost> {
    keys: Vec<K>,
    residual: R,
    noise: Option<N>,
    robust: Option<C>,
    _phantom: std::marker::PhantomData<V>,
}

impl<K: Key, V: Variable, R: Residual<V>, N: NoiseModel, C: RobustCost>
    FactorGeneric<K, V, R, N, C>
{
    #[allow(clippy::new_ret_no_self)]
    pub fn new(keys: Vec<K>, residual: impl Into<R>) -> FactorFactory<K, V, R, N, C> {
        // TODO: Need to check # of keys and residual vars are the same
        let residual = residual.into();
        FactorFactory {
            keys,
            residual,
            noise: None,
            robust: None,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn error(&self, values: &Values<K, V>) -> dtype {
        let r = self.residual.residual(values, &self.keys);
        let r = self.noise.whiten_vec(&r);
        let norm2 = r.norm_squared();
        self.robust.loss(norm2)
    }

    pub fn linearize(&self, values: &Values<K, V>) -> LinearFactor<K> {
        // Compute residual and jacobian
        let DiffResult { value: r, diff: a } = self.residual.residual_jacobian(values, &self.keys);

        // Whiten residual and jacobian
        let r = self.noise.whiten_vec(&r);
        let a = self.noise.whiten_mat(&a);

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

impl<K: Key, V: Variable, R: Residual<V>, N: NoiseModel, C: RobustCost> FactorFactory<K, V, R, N, C>
where
    N: From<GaussianNoise>,
    C: From<L2>,
{
    pub fn set_noise(mut self, noise: impl Into<N>) -> Self {
        let noise = noise.into();
        assert_eq!(
            noise.dim(),
            self.residual.dim(),
            "Noise dimension must match residual dimension"
        );
        self.noise = Some(noise);
        self
    }

    pub fn set_robust(mut self, robust: impl Into<C>) -> Self {
        self.robust = Some(robust.into());
        self
    }

    pub fn build(self) -> FactorGeneric<K, V, R, N, C> {
        let d = self.residual.dim();
        // TODO: Should we cater to situations where noise or robustness has a different default?
        FactorGeneric {
            keys: self.keys,
            residual: self.residual,
            noise: self
                .noise
                .unwrap_or_else(|| GaussianNoise::identity(d).into()),
            robust: self.robust.unwrap_or_else(|| L2.into()),
            _phantom: std::marker::PhantomData,
        }
    }
}

// Type alias to make life easier
use crate::bundle::{Bundle, DefaultBundle};
pub type Factor<B = DefaultBundle> = FactorGeneric<
    <B as Bundle>::Key,
    <B as Bundle>::Variable,
    <B as Bundle>::Residual,
    <B as Bundle>::Noise,
    <B as Bundle>::Robust,
>;

#[cfg(test)]
mod tests {
    use nalgebra::dvector;

    use crate::{
        containers::X,
        linalg::{NumericalDiff, Vector3},
        residuals::{BetweenResidual, PriorResidual},
        robust::GemanMcClure,
    };
    use matrixcompare::assert_matrix_eq;

    use super::*;

    // Gets a little tricky with the robust cost
    #[test]
    fn linearize_a() {
        let prior = Vector3::new(1.0, 2.0, 3.0);
        let x = <Vector3 as Variable>::identity();

        let keys = vec![X(0)];
        let residual = PriorResidual::new(&prior);
        let noise = GaussianNoise::from_diag_sigma(&dvector![1e-1, 2e-1, 3e-1]);
        let robust = GemanMcClure::default();

        let factor = Factor::<DefaultBundle>::new(keys, residual)
            .set_noise(noise)
            .set_robust(robust)
            .build();

        let f = |x: Vector3| {
            let mut values = Values::new();
            values.insert(X(0), x);
            factor.error(&values)
        };

        let mut values = Values::new();
        values.insert(X(0), x);

        let linear = factor.linearize(&values);
        let grad_got = -linear.a.mat().transpose() * linear.b;
        println!("Received {:}", grad_got);

        let grad_num = NumericalDiff::<6>::gradient_1(f, &x).diff;
        println!("Expected {:}", grad_num);

        assert_matrix_eq!(grad_got, grad_num, comp = abs, tol = 1e-6);
    }

    #[test]
    fn linearize_block() {
        let bet = Vector3::new(1.0, 2.0, 3.0);
        let x = <Vector3 as Variable>::identity();

        let keys = vec![X(0), X(1)];
        let residual = BetweenResidual::new(&bet);
        let noise = GaussianNoise::from_diag_sigma(&dvector![1e-1, 2e-1, 3e-1]);
        let robust = GemanMcClure::default();

        let factor = Factor::<DefaultBundle>::new(keys, residual)
            .set_noise(noise)
            .set_robust(robust)
            .build();

        let mut values = Values::new();
        values.insert(X(0), x);
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
