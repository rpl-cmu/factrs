use crate::variables::Variable;
use nalgebra::SVector;

impl<const N: usize> Variable for SVector<f64, N> {
    type TangentVec = SVector<f64, N>;
    const DIM: usize = N;

    fn identity() -> Self {
        Self::TangentVec::zeros()
    }

    fn inverse(&self) -> Self {
        -self
    }

    fn oplus(&self, delta: &Self::TangentVec) -> Self {
        self + delta
    }

    fn ominus(&self, other: &Self) -> Self::TangentVec {
        self - other
    }
}

pub type Vector1 = SVector<f64, 1>;
pub type Vector2 = SVector<f64, 2>;
pub type Vector3 = SVector<f64, 3>;
pub type Vector4 = SVector<f64, 4>;
pub type Vector5 = SVector<f64, 5>;
pub type Vector6 = SVector<f64, 6>;
pub type Vector7 = SVector<f64, 7>;
pub type Vector8 = SVector<f64, 8>;
pub type Vector9 = SVector<f64, 9>;
pub type Vector10 = SVector<f64, 10>;
