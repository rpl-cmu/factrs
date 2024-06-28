use crate::{
    containers::{GraphOrder, Idx, Order},
    dtype,
    linalg::DiffResult,
    linear::LinearFactor,
};
use faer::sparse::{SparseColMat, SymbolicSparseColMat};
use faer_ext::IntoFaer;

use super::LinearValues;

#[derive(Default)]
pub struct LinearGraph {
    factors: Vec<LinearFactor>,
}

impl LinearGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_vec(factors: Vec<LinearFactor>) -> Self {
        Self { factors }
    }

    pub fn add_factor(&mut self, factor: LinearFactor) {
        self.factors.push(factor);
    }

    pub fn error(&self, values: &LinearValues) -> dtype {
        self.factors.iter().map(|f| f.error(values)).sum()
    }

    // TODO: This is identical for nonlinear case, is there a way we can reduce code reuse?
    pub fn sparsity_pattern(&self, order: Order) -> GraphOrder {
        let total_rows = self.factors.iter().map(|f| f.dim_out()).sum();
        let total_columns = order.dim();

        let mut indices = Vec::<(usize, usize)>::new();

        let _ = self.factors.iter().fold(0, |row, f| {
            f.keys.iter().for_each(|key| {
                (0..f.dim_out()).for_each(|i| {
                    let Idx {
                        idx: col,
                        dim: col_dim,
                    } = order.get(key).unwrap();
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

    pub fn residual_jacobian(
        &self,
        graph_order: &GraphOrder,
    ) -> DiffResult<faer::Mat<dtype>, SparseColMat<usize, dtype>> {
        // Create the residual vector
        let total_rows = self.factors.iter().map(|f| f.dim_out()).sum();
        let mut r = faer::Mat::zeros(total_rows, 1);
        let _ = self.factors.iter().fold(0, |row, f| {
            r.subrows_mut(row, f.dim_out())
                .copy_from(&f.b.view_range(.., ..).into_faer());
            row + f.dim_out()
        });

        // Create the jacobian matrix
        let mut values: Vec<dtype> = Vec::new();
        // Iterate over all factors
        let _ = self.factors.iter().fold(0, |row, f| {
            // Iterate over keys
            (0..f.keys.len()).for_each(|idx| {
                // Iterate over rows, then column elements
                f.a.get_block(idx).row_iter().for_each(|r| {
                    r.iter().for_each(|val| {
                        values.push(*val);
                    });
                });
            });
            row + f.dim_out()
        });

        let jac = SparseColMat::new_from_order_and_values(
            graph_order.sparsity_pattern.clone(),
            &graph_order.sparsity_order,
            values.as_slice(),
        )
        .unwrap();

        DiffResult {
            value: r,
            diff: jac,
        }
    }
}

#[cfg(test)]
mod test {
    use ahash::HashMap;
    use faer_ext::IntoNalgebra;
    use matrixcompare::assert_matrix_eq;

    use crate::{
        containers::{Idx, X},
        linalg::{MatrixBlock, MatrixX, VectorX},
    };

    use super::*;

    #[test]
    fn residual_jacobian() {
        // Make a linear graph
        let mut graph = LinearGraph::new();

        // Make a handful of factors
        let a1 = MatrixX::from_fn(2, 2, |i, j| (i + j) as dtype);
        let block1 = MatrixBlock::new(a1, vec![0]);
        let b1 = VectorX::from_fn(2, |i, j| (i + j) as dtype);
        graph.add_factor(LinearFactor::new(vec![X(1)], block1.clone(), b1.clone()));

        let a2 = MatrixX::from_fn(3, 5, |i, j| (i + j) as dtype);
        let block2 = MatrixBlock::new(a2, vec![0, 2]);
        let b2 = VectorX::from_fn(3, |_, _| 5.0);
        graph.add_factor(LinearFactor::new(
            vec![X(0), X(2)],
            block2.clone(),
            b2.clone(),
        ));

        // Make fake ordering
        let mut map = HashMap::default();
        map.insert(X(0), Idx { idx: 0, dim: 2 });
        map.insert(X(1), Idx { idx: 2, dim: 2 });
        map.insert(X(2), Idx { idx: 4, dim: 3 });
        let order = Order::new(map);

        // Compute the residual and jacobian
        let graph_order = graph.sparsity_pattern(order);
        let DiffResult { value, diff } = graph.residual_jacobian(&graph_order);
        let value = value.as_ref().into_nalgebra().clone_owned();
        let diff = diff.to_dense().as_ref().into_nalgebra().clone_owned();

        println!("Value: {}", value);
        println!("Diff: {}", diff);

        // Check the residual
        assert_matrix_eq!(b1, value.rows(0, 2), comp = float);
        assert_matrix_eq!(b2, value.rows(2, 3), comp = float);

        // Check the jacobian
        assert_matrix_eq!(block1.get_block(0), diff.view((0, 2), (2, 2)), comp = float);
        assert_matrix_eq!(block2.get_block(0), diff.view((2, 0), (3, 2)), comp = float);
        assert_matrix_eq!(block2.get_block(1), diff.view((2, 4), (3, 3)), comp = float);
    }
}
