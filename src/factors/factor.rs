use super::LinearFactor;
use crate::containers::Values;
use crate::dtype;
use crate::linalg::VectorX;
use crate::noise::{GaussianNoise, NoiseModel};
use crate::residuals::Residual;
use crate::robust::{RobustCost, L2};
use crate::traits::{Key, Variable};

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
        let residual = residual.into();
        FactorFactory {
            keys,
            residual,
            noise: None,
            robust: None,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn error_vec(&self, values: &Values<K, V>) -> VectorX {
        let r = self.residual.residual(values, &self.keys);
        let r = self.noise.whiten_vec(&r);
        // Divide by sqrt 2?
        self.robust.weight_vec(&r)
    }

    pub fn error_scalar(&self, values: &Values<K, V>) -> dtype {
        let r = self.residual.residual(values, &self.keys);
        let r = self.noise.whiten_vec(&r);
        let norm2 = r.norm_squared();
        norm2 * self.robust.weight(norm2) / 2.0
    }

    pub fn linearize(&self, values: &Values<K, V>) -> LinearFactor<K> {
        let (r, h) = self.residual.residual_jacobian(values, &self.keys);
        let norm2 = r.norm_squared();
        let weight = self.robust.weight(norm2).sqrt();
        let a = weight * self.noise.whiten_mat(&h);
        let b = -weight * self.noise.whiten_vec(&r);
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
        linalg::Vector3, residuals::PriorResidual, robust::GemanMcClure, utils::num_derivative_11,
        variables::X,
    };

    use super::*;

    // TODO: Go through the math to figure out how to test A matrix
    // Gets a little tricky with the robust cost
    #[test]
    fn linearize_a() {
        let x = <Vector3 as Variable>::identity();
        let prior = Vector3::new(1.0, 2.0, 3.0);

        let keys = vec![X(0)];
        let residual = PriorResidual::new(&prior);
        let noise = GaussianNoise::from_diag_sigma(&dvector![4.0, 5.0, 6.0]);
        let robust = GemanMcClure::default();

        let factor = Factor::<DefaultBundle>::new(keys, residual)
            .set_noise(noise)
            .set_robust(robust)
            .build();

        let f = |x: Vector3| {
            let mut values = Values::new();
            values.insert(X(0), x);
            factor.error_vec(&values)
        };

        let mut values = Values::new();
        values.insert(X(0), x);

        let linear = factor.linearize(&values);
        println!("{:}", linear.a);

        let jac = num_derivative_11(f, x);
        println!("{:}", jac);

        panic!();
    }

    #[test]
    fn linearize_b() {}
}
