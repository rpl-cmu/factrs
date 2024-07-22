use std::{
    fmt::{Debug, Display},
    ops::Index,
};

use crate::{
    dtype,
    linalg::{
        AllocatorBuffer,
        Const,
        DefaultAllocator,
        DimName,
        DualAllocator,
        DualVector,
        Numeric,
        Vector,
        VectorDim,
        VectorViewX,
        VectorX,
    },
    tag_variable,
    variables::Variable,
};

tag_variable!(
    VectorVar<1>,
    VectorVar<2>,
    VectorVar<3>,
    VectorVar<4>,
    VectorVar<5>,
    VectorVar<6>,
);

// ------------------------- Our needs ------------------------- //
/// Newtype wrapper around nalgebra::Vector
///
/// We create a newtype specifically for vectors we're estimating over due to,
/// 1 - So we can manually implement Debug/Display
/// 2 - Overcome identity issues with the underlying Vector type
/// 3 - Impl Into\<Rerun types\>
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VectorVar<const N: usize, D: Numeric = dtype>(pub Vector<N, D>);

impl<const N: usize, D: Numeric> Variable<D> for VectorVar<N, D> {
    type Dim = Const<N>;
    type Alias<DD: Numeric> = VectorVar<N, DD>;

    fn identity() -> Self {
        VectorVar(Vector::zeros())
    }

    fn inverse(&self) -> Self {
        VectorVar(-self.0)
    }

    fn compose(&self, other: &Self) -> Self {
        VectorVar(self.0 + other.0)
    }

    fn exp(delta: VectorViewX<D>) -> Self {
        VectorVar(Vector::from_iterator(delta.iter().cloned()))
    }

    fn log(&self) -> VectorX<D> {
        VectorX::from_iterator(Self::DIM, self.0.iter().cloned())
    }

    fn dual_convert<DD: Numeric>(other: &Self::Alias<dtype>) -> Self::Alias<DD> {
        VectorVar(other.0.map(|x| x.into()))
    }

    // Mostly unncessary, but avoids having to convert VectorX to static vector
    fn dual_setup<NN: DimName>(idx: usize) -> Self::Alias<DualVector<NN>>
    where
        AllocatorBuffer<NN>: Sync + Send,
        DefaultAllocator: DualAllocator<NN>,
        DualVector<NN>: Copy,
    {
        let n = VectorDim::<NN>::zeros().shape_generic().0;
        let mut tv = Vector::<N, DualVector<NN>>::zeros();
        for (i, tvi) in tv.iter_mut().enumerate() {
            tvi.eps = num_dual::Derivative::derivative_generic(n, Const::<1>, idx + i);
        }
        VectorVar(tv)
    }
}

// TODO: New methods for each size
macro_rules! impl_vector_new {
    ($($num:literal, [$($args:ident),*]);* $(;)?) => {$(
        impl<D: Numeric> VectorVar<$num, D> {
            pub fn new($($args: D),*) -> Self {
                VectorVar(Vector::<$num, D>::new($($args),*))
            }
        }
    )*};
}

impl_vector_new!(
    1, [x];
    2, [x, y];
    3, [x, y, z];
    4, [x, y, z, w];
    5, [x, y, z, w, a];
    6, [x, y, z, w, a, b];
);

impl<const N: usize, D: Numeric> From<Vector<N, D>> for VectorVar<N, D> {
    fn from(v: Vector<N, D>) -> Self {
        VectorVar(v)
    }
}

impl<const N: usize, D: Numeric> From<VectorVar<N, D>> for Vector<N, D> {
    fn from(v: VectorVar<N, D>) -> Self {
        v.0
    }
}

impl<const N: usize, D: Numeric> Display for VectorVar<N, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Vector{}(", N)?;
        for (i, x) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{:.3}", x)?;
        }
        write!(f, ")")
    }
}

impl<const N: usize, D: Numeric> Debug for VectorVar<N, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(self, f)
    }
}

impl<const N: usize, D: Numeric> Index<usize> for VectorVar<N, D> {
    type Output = D;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

/// 1D Vector Variable
pub type VectorVar1<D = dtype> = VectorVar<1, D>;
/// 2D Vector Variable
pub type VectorVar2<D = dtype> = VectorVar<2, D>;
/// 3D Vector Variable
pub type VectorVar3<D = dtype> = VectorVar<3, D>;
/// 4D Vector Variable
pub type VectorVar4<D = dtype> = VectorVar<4, D>;
/// 5D Vector Variable
pub type VectorVar5<D = dtype> = VectorVar<5, D>;
/// 6D Vector Variable
pub type VectorVar6<D = dtype> = VectorVar<6, D>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_variable;

    // Be lazy and only test Vector6 - others should work the same
    test_variable!(VectorVar6);
}
