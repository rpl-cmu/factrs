use crate::dtype;
use crate::linalg::{
    Const, Dim, DualNum, DualVec, Dyn, MatrixDim, MatrixViewDim, VectorViewX, VectorX,
};
use nalgebra as na;

use std::fmt::{Debug, Display};

pub trait Variable<D: DualNum = dtype>: Clone + Sized + Display + Debug {
    const DIM: usize;
    type Dual: Variable<DualVec>;

    // Group operations
    fn identity() -> Self;
    fn inverse(&self) -> Self;
    fn compose(&self, other: &Self) -> Self;
    fn exp(delta: VectorViewX<D>) -> Self; // trivial if linear (just itself)
    fn log(&self) -> VectorX<D>; // trivial if linear (just itself)

    // Conversion to dual space
    fn dual_self(&self) -> Self::Dual;

    // Helpers for enum
    fn dim(&self) -> usize {
        Self::DIM
    }
    fn identity_enum(&self) -> Self {
        Self::identity()
    }

    // Moves to and from vector space
    fn oplus(&self, delta: VectorViewX<D>) -> Self {
        if cfg!(feature = "left") {
            Self::exp(delta).compose(self)
        } else {
            self.compose(&Self::exp(delta))
        }
    }
    fn ominus(&self, other: &Self) -> VectorX<D> {
        if cfg!(feature = "left") {
            self.compose(&other.inverse()).log()
        } else {
            other.inverse().compose(self).log()
        }
    }
    fn minus(&self, other: &Self) -> Self {
        other.inverse().compose(self)
    }

    // Create tangent vector w/ duals set up properly
    fn dual_tangent(&self, idx: usize, total: usize) -> VectorX<DualVec> {
        let mut tv: VectorX<DualVec> = VectorX::zeros(self.dim());
        for (i, tvi) in tv.iter_mut().enumerate() {
            tvi.eps = num_dual::Derivative::derivative_generic(Dyn(total), Const::<1>, idx + i);
        }
        tv
    }
    // Applies the tangent vector in dual space
    fn dual(&self, idx: usize, total: usize) -> Self::Dual {
        self.dual_self()
            .oplus(self.dual_tangent(idx, total).as_view())
    }
}

pub trait MatrixLieGroup<D: DualNum = dtype>: Variable<D>
where
    na::DefaultAllocator: na::allocator::Allocator<D, Self::TangentDim, Self::TangentDim>,
    na::DefaultAllocator: na::allocator::Allocator<D, Self::MatrixDim, Self::MatrixDim>,
    na::DefaultAllocator: na::allocator::Allocator<D, Self::VectorDim, Self::TangentDim>,
    na::DefaultAllocator: na::allocator::Allocator<D, Self::TangentDim, Const<1>>,
    na::DefaultAllocator: na::allocator::Allocator<D, Self::VectorDim, Const<1>>,
{
    type TangentDim: Dim;
    type MatrixDim: Dim;
    type VectorDim: Dim;

    fn adjoint(&self) -> MatrixDim<Self::TangentDim, Self::TangentDim, D>;

    fn hat(
        xi: MatrixViewDim<'_, Self::TangentDim, Const<1>, D>,
    ) -> MatrixDim<Self::MatrixDim, Self::MatrixDim, D>;

    fn vee(
        xi: MatrixViewDim<'_, Self::MatrixDim, Self::MatrixDim, D>,
    ) -> MatrixDim<Self::TangentDim, Const<1>, D>;

    fn hat_swap(
        xi: MatrixViewDim<'_, Self::VectorDim, Const<1>, D>,
    ) -> MatrixDim<Self::VectorDim, Self::TangentDim, D>;

    fn apply(
        &self,
        v: MatrixViewDim<'_, Self::VectorDim, Const<1>, D>,
    ) -> MatrixDim<Self::VectorDim, Const<1>, D>;

    fn to_matrix(&self) -> MatrixDim<Self::MatrixDim, Self::MatrixDim, D>;

    fn from_matrix(mat: MatrixViewDim<'_, Self::MatrixDim, Self::MatrixDim, D>) -> Self;
}
