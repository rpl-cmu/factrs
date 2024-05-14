use crate::variables::VectorD;
use std::fmt::Display;
use std::ops::Mul;

pub trait Variable: Clone + Sized + Display {
    const DIM: usize;

    fn dim(&self) -> usize {
        Self::DIM
    }

    fn identity() -> Self;

    fn identity_enum(&self) -> Self {
        Self::identity()
    }

    fn inverse(&self) -> Self;

    fn oplus(&self, delta: &VectorD) -> Self;

    fn ominus(&self, other: &Self) -> VectorD;
}

pub trait LieGroup: Variable + Mul {
    fn exp(xi: &VectorD) -> Self;

    fn log(&self) -> VectorD;
}
