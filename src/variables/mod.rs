// use ahash::AHashMap;
use nalgebra::{DVector, SVector};
use std::ops::Mul;

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
pub type VectorD = DVector<f64>;

// ------------------------- Import all variable types ------------------------- //
pub trait Variable: Sized + Clone {
    const DIM: usize;

    fn dim() -> usize {
        Self::DIM
    }

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

// mod values;
// pub use values::Values;

mod variable_enum;
pub use variable_enum::VariableEnum;

pub mod so3;
pub use so3::SO3;

pub mod vector;
