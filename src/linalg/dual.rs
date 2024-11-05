use nalgebra::{allocator::Allocator, Const};

use super::{Dim, Matrix, RealField};
use crate::dtype;

/// Wrapper for all properties needed for dual numbers
pub trait Numeric: RealField + num_dual::DualNum<dtype> + From<dtype> + Copy {}
impl<G: RealField + num_dual::DualNum<dtype> + From<dtype> + Copy> Numeric for G {}

pub type DualVector<N> = num_dual::DualVec<dtype, dtype, N>;
pub type DualScalar = num_dual::Dual<dtype, dtype>;

/// Make allocator binds easier for dual numbers
pub trait DualAllocator<N: Dim>:
    Allocator<N> + Allocator<Const<1>, N> + Allocator<N, Const<1>> + Allocator<N, N>
{
}

impl<
        N: Dim,
        T: Allocator<N> + Allocator<Const<1>, N> + Allocator<N, Const<1>> + Allocator<N, N>,
    > DualAllocator<N> for T
{
}

// TODO: Expand on this instead of including in Variable??
pub trait DualConvert {
    type Alias<D: Numeric>;
    fn dual_convert<D: Numeric>(other: &Self::Alias<dtype>) -> Self::Alias<D>;
}

impl<const R: usize, const C: usize, T: Numeric> DualConvert for Matrix<R, C, T> {
    type Alias<D: Numeric> = Matrix<R, C, D>;
    fn dual_convert<D: Numeric>(other: &Self::Alias<dtype>) -> Self::Alias<D> {
        other.map(|x| x.into())
    }
}
