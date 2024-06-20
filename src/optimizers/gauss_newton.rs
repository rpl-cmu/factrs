use faer_ext::IntoNalgebra;

use crate::{
    containers::{GraphGeneric, Key, Order, Values},
    linalg::DiffResult,
    linear::{LinearSolver, LinearValues},
    noise::NoiseModel,
    residuals::Residual,
    robust::RobustCost,
    variables::Variable,
};

use super::{Optimizer, OptimizerParams};

#[derive(Default)]
pub struct GaussNewton;

impl Optimizer for GaussNewton {
    fn iterate<
        K: Key,
        V: Variable,
        R: Residual<V>,
        N: NoiseModel,
        C: RobustCost,
        S: LinearSolver,
    >(
        params: &mut OptimizerParams<S>,
        graph: &GraphGeneric<K, V, R, N, C>,
        values: &mut Values<K, V>,
    ) {
        // Make an ordering
        let order = Order::from_values(values);

        // Solve the linear system
        let linear_graph = graph.linearize(values);
        let DiffResult { value: r, diff: j } = linear_graph.residual_jacobian(&order);

        // Solve Ax = b
        let delta = params
            .solver
            .solve(&j, &r)
            .as_ref()
            .into_nalgebra()
            .column(0)
            .clone_owned();

        // Update the values
        let dx = LinearValues::from_order_and_values(order, delta);
        values.oplus_mut(&dx);
    }
}
