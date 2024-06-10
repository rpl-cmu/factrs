use crate::dtype;
use crate::linalg::{Dyn, RealField, VectorX};
use std::{cmp, fmt, hash};

// Dual field
pub trait DualNum:
    RealField + num_dual::DualNum<dtype> + Into<num_dual::DualVec<dtype, dtype, Dyn>>
{
}
impl<G: RealField + num_dual::DualNum<dtype> + Into<num_dual::DualVec<dtype, dtype, Dyn>>> DualNum
    for G
{
}
pub type DualVec = num_dual::DualVec<dtype, dtype, Dyn>;

// Variable
mod variable;
pub use variable::{LieGroup, Variable};

// Key
pub trait Key: cmp::Eq + cmp::PartialEq + hash::Hash + fmt::Display + Clone {}
impl<T: cmp::Eq + cmp::PartialEq + hash::Hash + fmt::Display + Clone> Key for T {}

// Residual
use crate::residuals::Residual;

// ------------------------- TODO: Rest of this is all WIP ------------------------- //
// Holds the enums optimize over
// This eliminates any need for dynamic dispatch
pub trait Bundle: Sized {
    type Key: Key;
    type Variable: Variable<dtype>;
    type Robust: RobustCost;
    type Noise: NoiseModel;
    type Residual: Residual<Self::Variable>;
}
pub trait NoiseModel: Sized {
    fn dim(&self) -> usize;

    fn whiten(&self, v: &VectorX) -> VectorX;
}

pub trait RobustCost: Sized {
    fn weight(&self, d2: dtype) -> dtype;

    fn weight_vector(&self, r: &VectorX) -> VectorX {
        r * self.weight(r.norm_squared()).sqrt()
    }
}
