use nalgebra::{allocator::Allocator, Const};

use super::{Dim, RealField};
use crate::dtype;

/// Wrapper for all properties needed for dual numbers
pub trait Numeric: RealField + num_dual::DualNum<dtype> + From<dtype> + Copy {}
impl<G: RealField + num_dual::DualNum<dtype> + From<dtype> + Copy> Numeric for G {}

pub type DualVector<N> = num_dual::DualVec<dtype, dtype, N>;
pub type DualScalar = num_dual::Dual<dtype, dtype>;

/// Make allocator binds easier for dual numbers
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
