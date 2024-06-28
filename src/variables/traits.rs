use crate::{
    dtype,
    linalg::{
        Const, DimName, DualNum, DualVec, Dyn, MatrixDim, MatrixViewDim, VectorViewX, VectorX,
    },
};

use downcast_rs::{impl_downcast, Downcast};
use std::fmt::{Debug, Display};

pub trait Variable<D: DualNum = dtype>: Clone + Sized + Display + Debug {
    type Dim: DimName;
    const DIM: usize = Self::Dim::USIZE;
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

    // Setup group element correctly using the tangent space
    fn dual_setup(idx: usize, total: usize) -> Self::Dual {
        let mut tv: VectorX<DualVec> = VectorX::zeros(Self::DIM);
        for (i, tvi) in tv.iter_mut().enumerate() {
            tvi.eps = num_dual::Derivative::derivative_generic(Dyn(total), Const::<1>, idx + i);
        }
        Self::Dual::exp(tv.as_view())
    }

    // Applies the tangent vector in dual space
    fn dual(&self, idx: usize, total: usize) -> Self::Dual {
        // Setups tangent vector -> exp, then we compose here
        let setup = Self::dual_setup(idx, total);
        if cfg!(feature = "left") {
            setup.compose(&self.dual_self())
        } else {
            self.dual_self().compose(&setup)
        }
    }
}

pub trait VariableSafe: Debug + Display + Downcast {
    fn clone_box(&self) -> Box<dyn VariableSafe>;

    fn dim(&self) -> usize;

    fn oplus_mut(&mut self, delta: VectorViewX);
}

impl<T: Variable + 'static> VariableSafe for T {
    fn clone_box(&self) -> Box<dyn VariableSafe> {
        Box::new((*self).clone())
    }

    fn dim(&self) -> usize {
        self.dim()
    }

    fn oplus_mut(&mut self, delta: VectorViewX) {
        *self = self.oplus(delta);
    }
}

impl_downcast!(VariableSafe);

impl Clone for Box<dyn VariableSafe> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

use nalgebra as na;

pub trait MatrixLieGroup<D: DualNum = dtype>: Variable<D>
where
    na::DefaultAllocator: na::allocator::Allocator<D, Self::TangentDim, Self::TangentDim>,
    na::DefaultAllocator: na::allocator::Allocator<D, Self::MatrixDim, Self::MatrixDim>,
    na::DefaultAllocator: na::allocator::Allocator<D, Self::VectorDim, Self::TangentDim>,
    na::DefaultAllocator: na::allocator::Allocator<D, Self::TangentDim, Const<1>>,
    na::DefaultAllocator: na::allocator::Allocator<D, Self::VectorDim, Const<1>>,
{
    type TangentDim: DimName;
    type MatrixDim: DimName;
    type VectorDim: DimName;

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
