use crate::dtype;
use crate::linalg::{Dyn, RealField};
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
