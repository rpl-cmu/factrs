// use ahash::AHashMap;
use nalgebra::{DVector, SVector};
use std::ops::Mul;

// ------------------------- Import all variable types ------------------------- //

pub trait Variable: Clone + Sized {
    const DIM: usize;

    fn identity() -> Self;

    fn inverse(&self) -> Self;

    fn oplus(&self, delta: &VectorD) -> Self;

    fn ominus(&self, other: &Self) -> VectorD;
}

pub trait LieGroup: Variable + Mul {
    fn exp(xi: &VectorD) -> Self;

    fn log(&self) -> VectorD;
}

mod key;
pub use key::*;

mod values;
pub use values::Values;

mod variable_enum;
pub use variable_enum::{DispatchableVariable, VariableEnum};

pub mod so3;
pub use so3::SO3;

pub mod se3;
pub use se3::SE3;

pub mod vector;
pub use vector::*;
