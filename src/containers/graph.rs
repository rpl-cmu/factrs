use crate::{
    dtype,
    factors::{Factor, LinearFactor},
    noise::NoiseModel,
    residuals::Residual,
    robust::RobustCost,
    traits::{Key, Variable},
};

use super::Values;

pub struct Graph<K: Key, V: Variable, R: Residual<V>, N: NoiseModel, C: RobustCost> {
    factors: Vec<Factor<K, V, R, N, C>>,
}

impl<K: Key, V: Variable, R: Residual<V>, N: NoiseModel, C: RobustCost> Graph<K, V, R, N, C> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_factor(&mut self, factor: Factor<K, V, R, N, C>) {
        self.factors.push(factor);
    }

    pub fn error(&self, values: &Values<K, V>) -> dtype {
        self.factors.iter().map(|f| f.error(values)).sum()
    }

    pub fn linearize(&self, values: &Values<K, V>) -> Vec<LinearFactor<K>> {
        self.factors.iter().map(|f| f.linearize(values)).collect()
    }
}

impl<K: Key, V: Variable, R: Residual<V>, N: NoiseModel, C: RobustCost> Default
    for Graph<K, V, R, N, C>
{
    fn default() -> Self {
        Self {
            factors: Vec::new(),
        }
    }
}
