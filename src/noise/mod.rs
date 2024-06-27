use crate::linalg::{DimName, MatrixViewX, MatrixX, VectorViewX, VectorX};
pub trait NoiseModel: Sized {
    type Dim: DimName;

    fn dim(&self) -> usize {
        Self::Dim::USIZE
    }

    fn whiten_vec(&self, v: VectorViewX) -> VectorX;

    fn whiten_mat(&self, m: MatrixViewX) -> MatrixX;
}

pub trait NoiseModelSafe {
    fn dim(&self) -> usize;

    fn whiten_vec(&self, v: VectorViewX) -> VectorX;

    fn whiten_mat(&self, m: MatrixViewX) -> MatrixX;
}

impl<T: NoiseModel> NoiseModelSafe for T {
    fn dim(&self) -> usize {
        self.dim()
    }

    fn whiten_vec(&self, v: VectorViewX) -> VectorX {
        self.whiten_vec(v)
    }

    fn whiten_mat(&self, m: MatrixViewX) -> MatrixX {
        self.whiten_mat(m)
    }
}

mod gaussian;
pub use gaussian::GaussianNoise;
