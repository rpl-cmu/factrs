use crate::dtype;
use std::{cmp, fmt, hash};

// Dual field
pub trait DualNum<T>: RealField + num_dual::DualNum<T> {}
impl<T, G: RealField + num_dual::DualNum<T>> DualNum<T> for G {}

// Variable
mod variable;
use nalgebra::RealField;
pub use variable::{LieGroup, Variable};

// Key
pub trait Key: cmp::Eq + cmp::PartialEq + hash::Hash + fmt::Display + Clone {}
impl<T: cmp::Eq + cmp::PartialEq + hash::Hash + fmt::Display + Clone> Key for T {}

use crate::variables::VectorD;

// ------------------------- TODO: Rest of this is all WIP ------------------------- //
// Holds the enums optimize over
// This eliminates any need for dynamic dispatch
pub trait Bundle {
    type Key: Key;
    type Variable: Variable<dtype>;
    type Robust;
    type Noise;
    type Residual;
}

pub trait Residual<B: Bundle>: Sized {
    const DIM: usize;

    fn dim(&self) -> usize {
        Self::DIM
    }

    fn residual_values(&self, v: &[&B::Variable]) -> VectorD;
}

pub trait NoiseModel {
    const DIM: usize;

    fn dim(&self) -> usize {
        Self::DIM
    }

    fn whiten(&self, v: &VectorD) -> VectorD;
}

pub trait RobustCost {
    fn cost(&self, v: &VectorD) -> f64;
}

struct Factor<B: Bundle> {
    keys: Vec<B::Key>,
    residual: B::Residual,
    noise: B::Noise,
    robust: B::Robust,
}
