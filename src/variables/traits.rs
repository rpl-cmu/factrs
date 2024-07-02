use std::fmt::{Debug, Display};

use downcast_rs::{impl_downcast, Downcast};

use crate::{
    dtype,
    linalg::{
        AllocatorBuffer,
        Const,
        DefaultAllocator,
        DimName,
        DualAllocator,
        DualVector,
        MatrixDim,
        MatrixViewDim,
        Numeric,
        VectorDim,
        VectorViewX,
        VectorX,
    },
};

pub trait Variable<D: Numeric = dtype>: Clone + Sized + Display + Debug {
    type Dim: DimName;
    const DIM: usize = Self::Dim::USIZE;
    type Alias<DD: Numeric>: Variable<DD>;

    // Group operations
    fn identity() -> Self;
    fn inverse(&self) -> Self;
    fn compose(&self, other: &Self) -> Self;
    fn exp(delta: VectorViewX<D>) -> Self; // trivial if linear (just itself)
    fn log(&self) -> VectorX<D>; // trivial if linear (just itself)

    // Conversion to dual space
    fn dual_convert<DD: Numeric>(other: &Self::Alias<dtype>) -> Self::Alias<DD>;

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
    fn dual_setup<N: DimName>(idx: usize) -> Self::Alias<DualVector<N>>
    where
        AllocatorBuffer<N>: Sync + Send,
        DefaultAllocator: DualAllocator<N>,
        DualVector<N>: Copy,
    {
        let mut tv: VectorX<DualVector<N>> = VectorX::zeros(Self::DIM);
        let n = VectorDim::<N>::zeros().shape_generic().0;
        for (i, tvi) in tv.iter_mut().enumerate() {
            tvi.eps = num_dual::Derivative::derivative_generic(n, Const::<1>, idx + i)
        }
        Self::Alias::<DualVector<N>>::exp(tv.as_view())
    }

    // Applies the tangent vector in dual space
    fn dual<N: DimName>(other: &Self::Alias<dtype>, idx: usize) -> Self::Alias<DualVector<N>>
    where
        AllocatorBuffer<N>: Sync + Send,
        DefaultAllocator: DualAllocator<N>,
        DualVector<N>: Copy,
    {
        // Setups tangent vector -> exp, then we compose here
        let setup = Self::dual_setup(idx);
        if cfg!(feature = "left") {
            setup.compose(&Self::dual_convert(other))
        } else {
            Self::dual_convert(other).compose(&setup)
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

pub trait MatrixLieGroup<D: Numeric = dtype>: Variable<D>
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
