use super::LinearFactor;
use crate::containers::Values;
use crate::dtype;
use crate::noise::{GaussianNoise, NoiseModel};
use crate::residuals::Residual;
use crate::robust::{RobustCost, L2};
use crate::traits::{Key, Variable};

pub struct Factor<K: Key, V: Variable, R: Residual<V>, N: NoiseModel, C: RobustCost> {
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

impl<K: Key, V: Variable, R: Residual<V>, N: NoiseModel, C: RobustCost> Factor<K, V, R, N, C> {
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

    pub fn error(&self, values: &Values<K, V>) -> dtype {
        let r = self.residual.residual(values, &self.keys);
        let r = self.noise.whiten_vec(&r);
        let norm2 = r.norm_squared();
        norm2 * self.robust.weight(norm2) / 2.0
    }

    pub fn linearize(&self, values: &Values<K, V>) -> LinearFactor<K> {
        let (r, h) = self.residual.residual_jacobian(values, &self.keys);
        let norm2 = r.norm_squared();
        let weight = self.robust.weight(norm2);
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

    pub fn build(self) -> Factor<K, V, R, N, C> {
        let d = self.residual.dim();
        // TODO: Should we cater to situations where noise or robustness has a different default?
        Factor {
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
