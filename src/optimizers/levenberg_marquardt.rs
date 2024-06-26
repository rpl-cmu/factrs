use std::ops::Mul;

use faer::scale;
use faer::sparse::SparseColMat;
use faer_ext::IntoNalgebra;

use crate::{
    containers::{Graph, Key, Order, Values},
    dtype,
    linalg::DiffResult,
    linear::{LinearSolver, LinearValues},
    noise::NoiseModel,
    residuals::Residual,
    robust::RobustCost,
    variables::Variable,
};

use super::{OptError, OptResult, Optimizer, OptimizerParams};

pub struct LevenParams {
    pub lambda_min: dtype,
    pub lambda_max: dtype,
    pub lambda_factor: dtype,
    pub diagonal_damping: bool,
}

impl Default for LevenParams {
    fn default() -> Self {
        Self {
            lambda_min: 0.0,
            lambda_max: 1e5,
            lambda_factor: 10.0,
            diagonal_damping: true,
        }
    }
}

pub struct LevenMarquardt<
    K: Key,
    V: Variable,
    R: Residual<V>,
    N: NoiseModel,
    C: RobustCost,
    S: LinearSolver,
> {
    graph: Graph<K, V, R, N, C>,
    solver: S,
    pub params_base: OptimizerParams,
    pub params_leven: LevenParams,
    lambda: dtype,
}

impl<K: Key, V: Variable, R: Residual<V>, N: NoiseModel, C: RobustCost, S: LinearSolver>
    Optimizer<K, V, R, N, C> for LevenMarquardt<K, V, R, N, C, S>
{
    fn new(graph: Graph<K, V, R, N, C>) -> Self {
        Self {
            graph,
            solver: S::default(),
            params_base: OptimizerParams::default(),
            params_leven: LevenParams::default(),
            lambda: 1e-5,
        }
    }

    fn graph(&self) -> &Graph<K, V, R, N, C> {
        &self.graph
    }

    fn params(&self) -> &OptimizerParams {
        &self.params_base
    }

    // TODO: Some form of logging of the lambda value
    // TODO: More sophisticated stopping criteria based on magnitude of the gradient
    fn step(&mut self, mut values: Values<K, V>) -> OptResult<K, V> {
        // Make an ordering
        let order = Order::from_values(&values);

        // Get the linear system
        let linear_graph = self.graph.linearize(&values);
        let DiffResult { value: r, diff: j } = linear_graph.residual_jacobian(&order);

        // Form A
        let jtj = j
            .as_ref()
            .transpose()
            .to_col_major()
            .unwrap()
            .mul(j.as_ref());

        // Form I
        let triplets_i = if self.params_leven.diagonal_damping {
            (0..jtj.ncols())
                .map(|i| (i as isize, i as isize, jtj[(i, i)]))
                .collect::<Vec<_>>()
        } else {
            (0..jtj.ncols())
                .map(|i| (i as isize, i as isize, 1.0))
                .collect::<Vec<_>>()
        };
        let i = SparseColMat::<usize, dtype>::try_new_from_nonnegative_triplets(
            jtj.ncols(),
            jtj.ncols(),
            &triplets_i,
        )
        .unwrap();

        // Form b
        let b = j.as_ref().transpose().mul(&r);

        let mut dx = LinearValues::zero_from_order(order.clone());
        let old_error = linear_graph.error(&dx);

        loop {
            // Make Ax = b
            let a = &jtj + (&i * scale(self.lambda));

            // Solve Ax = b
            let delta = self
                .solver
                .solve_symmetric(&a, &b)
                .as_ref()
                .into_nalgebra()
                .column(0)
                .clone_owned();
            dx = LinearValues::from_order_and_values(order.clone(), delta);

            // Update our cost
            let curr_error = linear_graph.error(&dx);

            if curr_error < old_error {
                break;
            }

            self.lambda *= self.params_leven.lambda_factor;
            if self.lambda > self.params_leven.lambda_max {
                return Err(OptError::FailedToStep);
            }
        }

        // Update the values
        values.oplus_mut(&dx);
        self.lambda /= self.params_leven.lambda_factor;
        if self.lambda < self.params_leven.lambda_min {
            self.lambda = self.params_leven.lambda_min;
        }
        Ok(values)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        containers::Symbol, linear::CholeskySolver, noise::NoiseEnum, residuals::ResidualEnum,
        robust::RobustEnum, test_optimizer, variables::VariableEnum,
    };

    test_optimizer!(LevenMarquardt<Symbol, VariableEnum, ResidualEnum, NoiseEnum, RobustEnum, CholeskySolver>);
}
