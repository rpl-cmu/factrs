use std::{cmp, fmt, hash};

// Variable
mod variable;
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
    type Variable: Variable;
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

pub trait Residual1<B: Bundle>: Residual<B>
where
    for<'a> &'a B::Variable: TryInto<Self::FIRST>,
{
    type FIRST: Variable;

    fn residual_values(&self, v: &[&B::Variable]) -> VectorD {
        let x1: Self::FIRST = v[0].try_into().unwrap();
        self.residual(&x1)
    }

    fn residual(&self, x1: &Self::FIRST) -> VectorD;
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
