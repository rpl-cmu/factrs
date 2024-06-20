use crate::containers::{Key, Order};
use crate::dtype;
use crate::linalg::DiffResult;
use crate::linear::LinearFactor;
use faer::sparse::SparseColMat;
use faer_ext::IntoFaer;

use super::LinearValues;

pub struct LinearGraph<K: Key> {
    factors: Vec<LinearFactor<K>>,
}

impl<K: Key> LinearGraph<K> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_vec(factors: Vec<LinearFactor<K>>) -> Self {
        Self { factors }
    }

    pub fn add_factor(&mut self, factor: LinearFactor<K>) {
        self.factors.push(factor);
    }

    pub fn error(&self, values: &LinearValues<K>) -> f64 {
        self.factors.iter().map(|f| f.error(values)).sum()
    }

    #[allow(unused_variables)]
    pub fn residual_jacobian(
        &self,
        order: &Order<K>,
    ) -> DiffResult<faer::Mat<dtype>, SparseColMat<usize, dtype>> {
        let total_rows = self.factors.iter().map(|f| f.dim()).sum();
        let total_columns = order.dim();

        // Create the residual vector
        let mut r = faer::Mat::zeros(total_rows, 1);
        let _ = self.factors.iter().fold(0, |row, f| {
            r.subrows_mut(row, f.dim())
                .copy_from(&f.b.view_range(.., ..).into_faer());
            row + f.dim()
        });

        // Create the jacobian matrix
        let mut jac: Vec<(usize, usize, dtype)> = Vec::new();
        // Iterate over all factors
        let _ = self.factors.iter().fold(0, |row, f| {
            // Iterate over keys
            f.keys.iter().enumerate().for_each(|(idx, key)| {
                let col = order.get(key).unwrap().idx;
                // Iterate over elements
                f.a.get_block(idx)
                    .row_iter()
                    .enumerate()
                    .for_each(|(i, r)| {
                        r.iter().enumerate().for_each(|(j, val)| {
                            jac.push((row + i, col + j, *val));
                        });
                    });
            });
            row + f.dim()
        });

        println!("Jacobian: {:?}", jac.len());

        let jac = SparseColMat::try_new_from_triplets(total_rows, total_columns, &jac)
            .expect("Failed to form sparse jacobian");

        DiffResult {
            value: r,
            diff: jac,
        }
    }
}

impl<K: Key> Default for LinearGraph<K> {
    fn default() -> Self {
        Self {
            factors: Vec::new(),
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
        let a1 = MatrixX::from_fn(2, 2, |i, j| (i + j) as f64);
        let block1 = MatrixBlock::new(a1, vec![0]);
        let b1 = VectorX::from_fn(2, |i, j| (i + j) as f64);
        graph.add_factor(LinearFactor::new(vec![X(1)], block1.clone(), b1.clone()));

        let a2 = MatrixX::from_fn(3, 5, |i, j| (i + j) as f64);
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
        let DiffResult { value, diff } = graph.residual_jacobian(&order);
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
