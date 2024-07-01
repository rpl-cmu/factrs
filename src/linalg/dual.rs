use nalgebra::{allocator::Allocator, Const};
use num_dual::Dual;

use super::{Dim, Dyn, RealField};
use crate::dtype;

// Setup dual num
pub trait Numeric: RealField + num_dual::DualNum<dtype> + From<dtype> {}
impl<G: RealField + num_dual::DualNum<dtype> + From<dtype>> Numeric for G {}

pub type DualVectorX = num_dual::DualVec<dtype, dtype, Dyn>;
pub type DualVector<const N: usize> = num_dual::DualVec<dtype, dtype, Const<N>>;
pub type DualVectorGeneric<N> = num_dual::DualVec<dtype, dtype, N>;
pub type DualScalar = num_dual::Dual<dtype, dtype>;

pub trait DualAllocator<N: Dim>:
    Allocator<dtype, N>
    + Allocator<dtype, Const<1>, N>
    + Allocator<dtype, N, Const<1>>
    + Allocator<dtype, N, N>
{
}

impl<
        N: Dim,
        T: Allocator<dtype, N>
            + Allocator<dtype, Const<1>, N>
            + Allocator<dtype, N, Const<1>>
            + Allocator<dtype, N, N>,
    > DualAllocator<N> for T
{
}
