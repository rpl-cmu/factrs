use faer_ext::IntoNalgebra;

use crate::{
    bundle::Bundle,
    containers::{GraphGeneric, Key, Order, Values},
    linalg::DiffResult,
    linear::{CholeskySolver, LinearSolver, LinearValues},
    noise::NoiseModel,
    residuals::Residual,
    robust::RobustCost,
    variables::Variable,
};

use super::{OptResult, Optimizer, OptimizerParams};

#[derive(Default)]
pub struct GaussNewton<
    K: Key,
    V: Variable,
    R: Residual<V>,
    N: NoiseModel,
    C: RobustCost,
    S: LinearSolver = CholeskySolver,
> {
    graph: GraphGeneric<K, V, R, N, C>,
    solver: S,
    pub params: OptimizerParams,
}

impl<K: Key, V: Variable, R: Residual<V>, N: NoiseModel, C: RobustCost, S: LinearSolver>
    Optimizer<K, V, R, N, C> for GaussNewton<K, V, R, N, C, S>
{
    fn new(graph: GraphGeneric<K, V, R, N, C>) -> Self {
        Self {
            graph,
            solver: S::default(),
            params: OptimizerParams::default(),
        }
    }

    fn graph(&self) -> &GraphGeneric<K, V, R, N, C> {
        &self.graph
    }

    fn params(&self) -> &OptimizerParams {
        &self.params
    }

    // TODO: Should probably have some form of error handling here
    fn step(&mut self, mut values: Values<K, V>) -> OptResult<K, V> {
        // Make an ordering
        let order = Order::from_values(&values);

        // Solve the linear system
        let linear_graph = self.graph.linearize(&values);
        let DiffResult { value: r, diff: j } = linear_graph.residual_jacobian(&order);

        // Solve Ax = b
        let delta = self
            .solver
            .solve_lst_sq(&j, &r)
            .as_ref()
            .into_nalgebra()
            .column(0)
            .clone_owned();

        // Update the values
        let dx = LinearValues::from_order_and_values(order, delta);
        values.oplus_mut(&dx);

        Ok(values)
    }
}

pub type GaussNewtonBundled<B, S = CholeskySolver> = GaussNewton<
    <B as Bundle>::Key,
    <B as Bundle>::Variable,
    <B as Bundle>::Residual,
    <B as Bundle>::Noise,
    <B as Bundle>::Robust,
    S,
>;

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        containers::Symbol, linear::CholeskySolver, noise::NoiseEnum, residuals::ResidualEnum,
        robust::RobustEnum, test_optimizer, variables::VariableEnum,
    };

    test_optimizer!(GaussNewton<Symbol, VariableEnum, ResidualEnum, NoiseEnum, RobustEnum, CholeskySolver>);
}
