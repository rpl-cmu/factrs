use crate::dtype;
use crate::linalg::{Dyn, RealField, VectorX};
use crate::noise::NoiseModel;
use crate::robust::RobustCost;
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
