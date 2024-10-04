use faer::{
    solvers::{Cholesky, SolverCore},
    sparse::SymbolicSparseColMat,
};
use faer_ext::IntoNalgebra;

use super::{Idx, Symbol, TypedSymbol, Values, ValuesOrder};
use crate::{
    containers::Factor,
    dtype,
    linalg::{DiffResult, MatrixBlock},
    linear::{LinearFactor, LinearGraph},
    residuals::LinearResidual,
    variables::VariableUmbrella,
};

/// Structure to represent a nonlinear factor graph
///
/// Main usage will be via `add_factor` to add new [factors](Factor) to the
/// graph. Also of note is the `linearize` function that returns a [linear (aka
/// Gaussian) factor graph](LinearGraph).
///
/// Since the graph represents a nonlinear least-squares problem, during
/// optimization it will be iteratively linearized about a set of variables and
/// solved iteratively.
///
/// ```
/// # use factrs::prelude::*;
/// # assign_symbols!(X: SO2);
/// # let factor = FactorBuilder::new1(PriorResidual::new(SO2::identity()), X(0)).build();
/// let mut graph = Graph::new();
/// graph.add_factor(factor);
/// ```
#[derive(Default, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Graph {
    factors: Vec<Factor>,
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            factors: Vec::with_capacity(capacity),
        }
    }

    pub fn add_factor(&mut self, factor: Factor) {
        self.factors.push(factor);
    }

    pub fn len(&self) -> usize {
        self.factors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.factors.is_empty()
    }

    pub fn error(&self, values: &Values) -> dtype {
        self.factors.iter().map(|f| f.error(values)).sum()
    }

    pub fn linearize(&self, values: &Values) -> LinearGraph {
        let factors = self.factors.iter().map(|f| f.linearize(values)).collect();
        LinearGraph::from_vec(factors)
    }

    pub fn sparsity_pattern(&self, order: ValuesOrder) -> GraphOrder {
        let total_rows = self.factors.iter().map(|f| f.dim_out()).sum();
        let total_columns = order.dim();

        let mut indices = Vec::<(usize, usize)>::new();

        let _ = self.factors.iter().fold(0, |row, f| {
            f.keys().iter().for_each(|key| {
                (0..f.dim_out()).for_each(|i| {
                    let Idx {
                        idx: col,
                        dim: col_dim,
                    } = order.get(*key).unwrap();
                    (0..*col_dim).for_each(|j| {
                        indices.push((row + i, col + j));
                    });
                });
            });
            row + f.dim_out()
        });

        let (sparsity_pattern, sparsity_order) =
            SymbolicSparseColMat::try_new_from_indices(total_rows, total_columns, &indices)
                .unwrap();
        GraphOrder {
            order,
            sparsity_pattern,
            sparsity_order,
        }
    }

    #[allow(non_snake_case)]
    pub fn marginalize<K, V>(self, sym: impl Symbol, values: Values) -> (Self, Values) {
        // TODO: order should only use subset of keys from to_marg
        let order = ValuesOrder::from_values_skip(sym, &values);
        let key = sym.into();
        let factors = self.factors;

        // Find all factors that contain the key
        let (to_marg, factors): (Vec<_>, Vec<_>) =
            factors.into_iter().partition(|f| f.keys().contains(&key));

        // Create a new graph with just the key in it
        let linear_graph = Graph { factors: to_marg }.linearize(&values);
        let DiffResult {
            diff: jac,
            value: r,
        } = linear_graph.residual_jacobian_dense(&order);

        // TODO: Double check I can do this with just a subset of the values
        // Find schur complement pieces
        let key_dim = order.get(key).unwrap().dim;
        let dim = order.dim();
        let jtj = jac.as_ref().transpose() * jac.as_ref();
        let jtr = jac.as_ref().transpose() * r.as_ref();
        let A = jtj.submatrix(0, 0, dim - key_dim, dim - key_dim);
        let B = jtj.submatrix(0, dim - key_dim, dim - key_dim, key_dim);
        let C = jtj.submatrix(dim - key_dim, 0, key_dim, dim - key_dim);
        let Cinv = Cholesky::try_new(C, faer::Side::Upper).unwrap().inverse();
        let w = jtr.submatrix(0, 0, dim - key_dim, 1);
        let z = jtr.submatrix(dim - key_dim, 0, key_dim, 1);

        // Compute the new values
        let jtj = A - B * &Cinv * B.transpose();
        let jtr = w - B * Cinv * z;
        let j = Cholesky::try_new(jtj.as_ref(), faer::Side::Upper)
            .unwrap()
            .compute_l();
        let jtinv = Cholesky::try_new(j.as_ref().transpose(), faer::Side::Lower)
            .unwrap()
            .inverse();
        let r = jtinv * jtr;

        // Create the factor
        let a = j.as_ref().into_nalgebra().clone_owned();
        let a = MatrixBlock::new(a, todo!());
        let b = r.as_ref().into_nalgebra().column(0).clone_owned();
        let prior = LinearFactor {
            keys: todo!(),
            a,
            b,
        };
        let prior = LinearResidual::new(prior, todo!()).into_factor();
        factors.push(prior);

        // Finally, remove the key from the values
        values.remove_raw(sym);

        (Graph { factors }, values)
    }
}

/// Simple structure to hold the order of the graph
///
/// Specifically this is used to cache linearization results such as the order
/// of the graph and the sparsity pattern of the Jacobian (allows use to avoid
/// resorting indices).
pub struct GraphOrder {
    // Contains the order of the variables
    pub order: ValuesOrder,
    // Contains the sparsity pattern of the jacobian
    pub sparsity_pattern: SymbolicSparseColMat<usize>,
    // Contains the order of values to put into the sparsity pattern
    pub sparsity_order: faer::sparse::ValuesOrder<usize>,
}
