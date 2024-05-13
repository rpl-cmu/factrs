use std::fmt::Display;
use std::ops::Mul;

// ------------------------- Import all variable types ------------------------- //

pub trait Variable: Clone + Sized + Display {
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

mod symbol;
pub use symbol::*;

mod values;
pub use values::{Key, Values, Var};

mod variable_enum;
pub use variable_enum::{DispatchableVariable, VariableEnum};

pub mod so3;
pub use so3::SO3;

pub mod se3;
pub use se3::SE3;

pub mod vector;
pub use vector::*;
