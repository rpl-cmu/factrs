use crate::variables::{Variable, VectorD};
use nalgebra::{DVector, SVector};

impl<const N: usize> Variable for SVector<f64, N> {
    const DIM: usize = N;

    fn identity() -> Self {
        Self::zeros()
    }

    fn inverse(&self) -> Self {
        -self
    }

    fn oplus(&self, delta: &VectorD) -> Self {
        self + delta
    }

    fn ominus(&self, other: &Self) -> VectorD {
        let diff = self - other;
        DVector::from_iterator(Self::DIM, diff.iter().cloned())
    }
}
