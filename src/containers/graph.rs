use crate::{
    dtype, factors::Factor, linear::LinearGraph,
};

use super::Values;

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
}

