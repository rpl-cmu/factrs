use faer_ext::IntoNalgebra;

use crate::{
    containers::{Graph, GraphOrder, Values, ValuesOrder},
    linalg::DiffResult,
    linear::{CholeskySolver, LinearSolver, LinearValues},
};

use super::{OptResult, Optimizer, OptimizerParams};

#[derive(Default)]
pub struct GaussNewton<S: LinearSolver = CholeskySolver> {
    graph: Graph,
    solver: S,
    pub params: OptimizerParams,
    // For caching computation between steps
    graph_order: Option<GraphOrder>,
}

impl<S: LinearSolver> Optimizer for GaussNewton<S> {
    fn new(graph: Graph) -> Self {
        Self {
            graph,
            solver: S::default(),
            params: OptimizerParams::default(),
            graph_order: None,
        }
    }

    fn graph(&self) -> &Graph {
        &self.graph
    }

    fn params(&self) -> &OptimizerParams {
        &self.params
    }

    fn init(&mut self, _values: &Values) {
        // TODO: Some way to manual specify how to computer ValuesOrder
        // Precompute the sparsity pattern
        self.graph_order = Some(
            self.graph
                .sparsity_pattern(ValuesOrder::from_values(_values)),
        );
    }

    fn step(&mut self, mut values: Values) -> OptResult {
        // Solve the linear system
        let linear_graph = self.graph.linearize(&values);
        let DiffResult { value: r, diff: j } =
            linear_graph.residual_jacobian(self.graph_order.as_ref().unwrap());

        // Solve Ax = b
        let delta = self
            .solver
            .solve_lst_sq(j.as_ref(), r.as_ref())
            .as_ref()
            .into_nalgebra()
            .column(0)
            .clone_owned();

        // Update the values
        let dx = LinearValues::from_order_and_values(
            self.graph_order.as_ref().unwrap().order.clone(),
            delta,
        );
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
