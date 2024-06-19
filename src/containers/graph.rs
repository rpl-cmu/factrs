use crate::{
    dtype, factors::FactorGeneric, linear::LinearGraph, noise::NoiseModel, residuals::Residual,
    robust::RobustCost, variables::Variable,
};

use super::{Key, Values};

use crate::bundle::{Bundle, DefaultBundle};
pub type Graph<B = DefaultBundle> = GraphGeneric<
    <B as Bundle>::Key,
    <B as Bundle>::Variable,
    <B as Bundle>::Residual,
    <B as Bundle>::Noise,
    <B as Bundle>::Robust,
>;

pub struct GraphGeneric<K: Key, V: Variable, R: Residual<V>, N: NoiseModel, C: RobustCost> {
    factors: Vec<FactorGeneric<K, V, R, N, C>>,
}

impl<K: Key, V: Variable, R: Residual<V>, N: NoiseModel, C: RobustCost>
    GraphGeneric<K, V, R, N, C>
{
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_factor(&mut self, factor: FactorGeneric<K, V, R, N, C>) {
        self.factors.push(factor);
    }

    pub fn error(&self, values: &Values<K, V>) -> dtype {
        self.factors.iter().map(|f| f.error(values)).sum()
    }

    pub fn linearize(&self, values: &Values<K, V>) -> LinearGraph<K> {
        let factors = self.factors.iter().map(|f| f.linearize(values)).collect();
        LinearGraph::from_vec(factors)
    }
}

impl<K: Key, V: Variable, R: Residual<V>, N: NoiseModel, C: RobustCost> Default
    for GraphGeneric<K, V, R, N, C>
{
    fn default() -> Self {
        Self {
            factors: Vec::new(),
        }
    }
}
