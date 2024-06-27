use faer_ext::IntoNalgebra;

use crate::{
    containers::{Graph, Order, Values},
    linalg::DiffResult,
    linear::{CholeskySolver, LinearSolver, LinearValues},
};

use super::{OptResult, Optimizer, OptimizerParams};

#[derive(Default)]
pub struct GaussNewton<S: LinearSolver = CholeskySolver> {
    graph: Graph,
    solver: S,
    pub params: OptimizerParams,
}

impl<S: LinearSolver> Optimizer for GaussNewton<S> {
    fn new(graph: Graph) -> Self {
        Self {
            graph,
            solver: S::default(),
            params: OptimizerParams::default(),
        }
    }

    fn graph(&self) -> &Graph {
        &self.graph
    }

    fn params(&self) -> &OptimizerParams {
        &self.params
    }

    fn step(&mut self, mut values: Values) -> OptResult {
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

#[cfg(test)]
mod test {
    use super::*;
    use crate::test_optimizer;

    test_optimizer!(GaussNewton);
}
