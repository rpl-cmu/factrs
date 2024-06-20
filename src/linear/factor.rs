use crate::containers::Key;
use crate::linalg::{MatrixBlock, VectorX};
use crate::linear::LinearValues;

pub struct LinearFactor<K: Key> {
    pub keys: Vec<K>,
    pub a: MatrixBlock,
    pub b: VectorX,
}
impl<K: Key> LinearFactor<K> {
    pub fn new(keys: Vec<K>, a: MatrixBlock, b: VectorX) -> Self {
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

    pub fn error(&self, vector: &LinearValues<K>) -> f64 {
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
