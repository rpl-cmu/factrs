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

pub trait Variable: Clone + Sized + Display + Debug {
    type D: Numeric;
    type Dim: DimName;
    const DIM: usize = Self::Dim::USIZE;
    type Alias<DD: Numeric>: Variable<D = DD>;

    // Group operations
    fn identity() -> Self;
    fn inverse(&self) -> Self;
    fn compose(&self, other: &Self) -> Self;
    fn exp(delta: VectorViewX<Self::D>) -> Self; // trivial if linear (just itself)
    fn log(&self) -> VectorX<Self::D>; // trivial if linear (just itself)

    // Conversion to dual space
    fn dual_convert<DD: Numeric>(other: &Self::Alias<dtype>) -> Self::Alias<DD>;

    // Helpers for enum
    fn dim(&self) -> usize {
        Self::DIM
    }

    // Moves to and from vector space
    fn oplus(&self, delta: VectorViewX<Self::D>) -> Self {
        if cfg!(feature = "left") {
            Self::exp(delta).compose(self)
        } else {
            self.compose(&Self::exp(delta))
        }
    }
    fn ominus(&self, other: &Self) -> VectorX<Self::D> {
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

#[cfg_attr(feature = "serde", typetag::serde(tag = "tag"))]
pub trait VariableSafe: Debug + Display + Downcast {
    fn clone_box(&self) -> Box<dyn VariableSafe>;

    fn dim(&self) -> usize;

    fn oplus_mut(&mut self, delta: VectorViewX);
}

impl<
        #[cfg(not(feature = "serde"))] T: Variable<D = dtype> + 'static,
        #[cfg(feature = "serde")] T: Variable<D = dtype> + 'static + crate::serde::Tagged,
    > VariableSafe for T
{
    fn clone_box(&self) -> Box<dyn VariableSafe> {
        Box::new((*self).clone())
    }

    fn dim(&self) -> usize {
        self.dim()
    }

    fn oplus_mut(&mut self, delta: VectorViewX) {
        *self = self.oplus(delta);
    }

    #[doc(hidden)]
    #[cfg(feature = "serde")]
    fn typetag_name(&self) -> &'static str {
        Self::TAG
    }

    #[doc(hidden)]
    #[cfg(feature = "serde")]
    fn typetag_deserialize(&self) {}
}

pub trait VariableUmbrella: VariableSafe + Variable<D = dtype, Alias<dtype> = Self> {}
impl<T: VariableSafe + Variable<D = dtype, Alias<dtype> = Self>> VariableUmbrella for T {}

impl_downcast!(VariableSafe);

impl Clone for Box<dyn VariableSafe> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

use nalgebra as na;

pub trait MatrixLieGroup: Variable
where
    na::DefaultAllocator: na::allocator::Allocator<Self::D, Self::TangentDim, Self::TangentDim>,
    na::DefaultAllocator: na::allocator::Allocator<Self::D, Self::MatrixDim, Self::MatrixDim>,
    na::DefaultAllocator: na::allocator::Allocator<Self::D, Self::VectorDim, Self::TangentDim>,
    na::DefaultAllocator: na::allocator::Allocator<Self::D, Self::TangentDim, Const<1>>,
    na::DefaultAllocator: na::allocator::Allocator<Self::D, Self::VectorDim, Const<1>>,
{
    type TangentDim: DimName;
    type MatrixDim: DimName;
    type VectorDim: DimName;

    fn adjoint(&self) -> MatrixDim<Self::TangentDim, Self::TangentDim, Self::D>;

    fn hat(
        xi: MatrixViewDim<'_, Self::TangentDim, Const<1>, Self::D>,
    ) -> MatrixDim<Self::MatrixDim, Self::MatrixDim, Self::D>;

    fn vee(
        xi: MatrixViewDim<'_, Self::MatrixDim, Self::MatrixDim, Self::D>,
    ) -> MatrixDim<Self::TangentDim, Const<1>, Self::D>;

    fn hat_swap(
        xi: MatrixViewDim<'_, Self::VectorDim, Const<1>, Self::D>,
    ) -> MatrixDim<Self::VectorDim, Self::TangentDim, Self::D>;

    fn apply(
        &self,
        v: MatrixViewDim<'_, Self::VectorDim, Const<1>, Self::D>,
    ) -> MatrixDim<Self::VectorDim, Const<1>, Self::D>;

    fn to_matrix(&self) -> MatrixDim<Self::MatrixDim, Self::MatrixDim, Self::D>;

    fn from_matrix(mat: MatrixViewDim<'_, Self::MatrixDim, Self::MatrixDim, Self::D>) -> Self;
}
