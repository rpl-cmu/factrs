use faer::sparse::SymbolicSparseColMat;

use crate::{dtype, factors::Factor, linear::LinearGraph};

use super::{Idx, Values, ValuesOrder};

#[derive(Default)]
pub struct Graph {
    factors: Vec<Factor>,
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_factor(&mut self, factor: Factor) {
        self.factors.push(factor);
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
}

pub struct GraphOrder {
    // Contains the order of the variables
    pub order: ValuesOrder,
    // Contains the sparsity pattern of the jacobian
    pub sparsity_pattern: SymbolicSparseColMat<usize>,
    // Contains the order of values to put into the sparsity pattern
    pub sparsity_order: faer::sparse::ValuesOrder<usize>,
}
