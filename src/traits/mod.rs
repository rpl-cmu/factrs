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
pub trait Bundle: Sized {
    type Key: Key;
    type Variable: Variable<dtype>;
    type Robust: RobustCost;
    type Noise: NoiseModel;
    type Residual: Residual<Self>;
}

pub fn unpack<V: Variable<dtype>, B: Bundle>(b: B::Variable) -> V
where
    B::Variable: TryInto<V>,
{
    b.try_into().unwrap_or_else(|_| {
        panic!(
            "Failed to convert {} to {} in residual",
            std::any::type_name::<B::Variable>(),
            std::any::type_name::<V>()
        )
    })
}

pub trait Residual<B: Bundle>: Sized {
    const DIM: usize;

    fn dim(&self) -> usize {
        Self::DIM
    }

    fn residual(&self, v: &[&B::Variable]) -> VectorD;
}

struct PriorResidual<V: Variable<dtype>> {
    prior: V,
}

impl<B: Bundle, V: Variable<dtype>> Residual<B> for PriorResidual<V>
where
    for<'a> &'a B::Variable: TryInto<V>,
    B::Variable: TryInto<V>,
{
    const DIM: usize = V::DIM;
    fn residual(&self, v: &[&B::Variable]) -> VectorD {
        let x1: V = unpack::<V, B>(v[0].clone());
        x1.ominus(&self.prior)
    }
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
