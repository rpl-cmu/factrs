use crate::{
    containers::Symbol,
    dtype,
    linalg::{MatrixBlock, VectorX},
    linear::LinearValues,
};

pub struct LinearFactor {
    pub keys: Vec<Symbol>,
    pub a: MatrixBlock,
    pub b: VectorX,
}
impl LinearFactor {
    pub fn new(keys: Vec<Symbol>, a: MatrixBlock, b: VectorX) -> Self {
        assert!(
            keys.len() == a.idx().len(),
            "Mismatch between keys and matrix blocks in LinearFactor::new"
        );
        assert!(
            a.mat().nrows() == b.len(),
            "Mismatch between matrix block and vector in LinearFactor::new"
        );
        Self { keys, a, b }
    }

    pub fn dim(&self) -> usize {
        self.b.len()
    }

    pub fn error(&self, vector: &LinearValues) -> dtype {
        let ax: VectorX = self
            .keys
            .iter()
            .enumerate()
            .map(|(idx, key)| {
                self.a.mul(
                    idx,
                    vector.get(key).expect("Missing key in LinearValues::error"),
                )
            })
            .sum();
        (ax - &self.b).norm_squared() / 2.0
    }
}
