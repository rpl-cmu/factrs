use crate::linalg::{DimName, MatrixX, VectorX};
pub trait NoiseModel: Sized {
    type Dim: DimName;

    fn dim(&self) -> usize {
        Self::Dim::USIZE
    }

    fn whiten_vec(&self, v: &VectorX) -> VectorX;

    fn whiten_mat(&self, m: &MatrixX) -> MatrixX;
}

pub trait NoiseModelSafe {
    fn dim(&self) -> usize;

    fn whiten_vec(&self, v: &VectorX) -> VectorX;

    fn whiten_mat(&self, m: &MatrixX) -> MatrixX;
}

impl<T: NoiseModel> NoiseModelSafe for T {
    fn dim(&self) -> usize {
        self.dim()
    }

    fn whiten_vec(&self, v: &VectorX) -> VectorX {
        self.whiten_vec(v)
    }

    fn whiten_mat(&self, m: &MatrixX) -> MatrixX {
        self.whiten_mat(m)
    }
}

mod gaussian;
pub use gaussian::GaussianNoise;
