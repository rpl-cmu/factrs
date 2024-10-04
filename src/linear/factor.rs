use crate::{
    containers::Key,
    dtype,
    linalg::{MatrixBlock, MatrixViewX, VectorX},
    linear::LinearValues,
};

/// Represents a linear (aka Gaussian) factor.
///
/// This is the linear equivalent of [Factor](crate::containers::Factor). It
/// consists of the relevant keys, a [MatrixBlock] A, and a [VectorX] b. Again,
/// this *shouldn't* ever need to be used by hand.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

    pub fn dim_in(&self) -> usize {
        self.a.mat().ncols()
    }

    pub fn dim_out(&self) -> usize {
        self.b.len()
    }

    pub fn jacobian(&self) -> MatrixViewX {
        self.a.mat()
    }

    pub fn residual(&self, vector: &LinearValues) -> VectorX {
        let ax: VectorX = self
            .keys
            .iter()
            .enumerate()
            .map(|(idx, key)| {
                self.a.mul(
                    idx,
                    vector
                        .get(*key)
                        .expect("Missing key in LinearValues::residual"),
                )
            })
            .sum();
        ax - &self.b
    }

    pub fn error(&self, vector: &LinearValues) -> dtype {
        self.residual(vector).norm_squared() / 2.0
    }
}
