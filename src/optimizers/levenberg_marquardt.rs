use std::ops::Mul;

use faer::sparse::SparseColMat;
use faer_ext::IntoNalgebra;

use crate::{
    containers::{GraphGeneric, Key, Order, Values},
    dtype,
    linalg::DiffResult,
    linear::{LinearSolver, LinearValues},
    noise::NoiseModel,
    residuals::Residual,
    robust::RobustCost,
    variables::Variable,
};

use super::{Optimizer, OptimizerParams};

pub struct LevenMarquardt {
    lambda: dtype,
}

impl Default for LevenMarquardt {
    fn default() -> Self {
        Self { lambda: 1e-3 }
    }
}

impl Optimizer for LevenMarquardt {
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

        // Get the linear system
        let linear_graph = graph.linearize(values);
        let DiffResult { value: r, diff: j } = linear_graph.residual_jacobian(&order);

        // Form A & b
        let jtj = j
            .as_ref()
            .transpose()
            .to_col_major()
            .unwrap()
            .mul(j.as_ref());
        let triplets = (0..jtj.ncols())
            .map(|i| (i as isize, i as isize, 1e-3))
            .collect::<Vec<_>>();
        let i = SparseColMat::<usize, dtype>::try_new_from_nonnegative_triplets(
            jtj.ncols(),
            jtj.ncols(),
            &triplets,
        )
        .unwrap();
        let a = &jtj + &i;
        let b = j.as_ref().transpose().mul(&r);

        // Solve Ax = b
        // TODO: We need to form J^T * J + lambda * diag(J^T * J) * delta = -J^T * r our selves here
        // TODO: This implies we need to let the linear solver know we'll be passing in a symmatric matrix already
        // Diff result gives us our J and -r
        let delta = params
            .solver
            .solve_symmetric(&a, &b)
            .as_ref()
            .into_nalgebra()
            .column(0)
            .clone_owned();

        // Update the values
        let dx = LinearValues::from_order_and_values(order, delta);
        values.oplus_mut(&dx);
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::test_optimizer;

    test_optimizer!(LevenMarquardt);
}
