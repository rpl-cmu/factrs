use std::ops::Mul;

pub trait Variable: Sized + Clone {
    const DIM: usize;
    type TangentVec;

    fn identity() -> Self;

    fn oplus(&self, delta: &Self::TangentVec) -> Self;

    fn ominus(&self, other: &Self) -> Self::TangentVec;

    fn inverse(&self) -> Self;
}

pub trait LieGroup: Variable + Mul {
    fn exp(xi: &Self::TangentVec) -> Self;

    fn log(&self) -> Self::TangentVec;
}

// ------------------------- Import all variable types ------------------------- //
mod key;
pub use key::*;

pub mod so3;
pub use so3::SO3;

pub mod vector;
pub use vector::*;
