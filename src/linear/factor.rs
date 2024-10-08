use crate::{
    containers::Key,
    dtype,
    linalg::{MatrixBlock, VectorX},
    linear::LinearValues,
};

/// Represents a linear (aka Gaussian) factor.
///
/// This is the linear equivalent of [Factor](crate::containers::Factor). It
/// consists of the relevant keys, a [MatrixBlock] A, and a [VectorX] b. Again,
/// this *shouldn't* ever need to be used by hand.
pub struct LinearFactor {
    pub keys: Vec<Key>,
    pub a: MatrixBlock,
    pub b: VectorX,
}
impl LinearFactor {
    pub fn new(keys: Vec<Key>, a: MatrixBlock, b: VectorX) -> Self {
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

    pub fn dim_out(&self) -> usize {
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
                    vector
                        .get(*key)
                        .expect("Missing key in LinearValues::error"),
                )
            })
            .sum();
        (ax - &self.b).norm_squared() / 2.0
    }
}
