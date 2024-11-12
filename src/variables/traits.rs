use std::fmt::{Debug, Display};

use downcast_rs::{impl_downcast, Downcast};

use crate::{
    dtype,
    linalg::{
        AllocatorBuffer, Const, DefaultAllocator, DimName, DualAllocator, DualVector, MatrixDim,
        MatrixViewDim, Numeric, VectorDim, VectorViewX, VectorX,
    },
};

/// Variable trait for Lie groups
///
/// All variables must implement this trait to be used in the optimization
/// algorithms. See [module level documentation](crate::variables) for more
/// details.
pub trait Variable<T: Numeric = dtype>: Clone + Sized + Display + Debug {
    /// Dimension of the Lie group / Tangent space
    type Dim: DimName;
    const DIM: usize = Self::Dim::USIZE;
    /// Alias for the type for dual conversion
    type Alias<TT: Numeric>: Variable<TT>;

    // Group operations
    /// Identity element of the group
    fn identity() -> Self;
    /// Inverse of the group element
    fn inverse(&self) -> Self;
    /// Composition of two group elements
    fn compose(&self, other: &Self) -> Self;
    /// Exponential map (trivial if a vectorspace)
    fn exp(delta: VectorViewX<T>) -> Self;
    /// Logarithm map (trivial if a vectorspace)
    fn log(&self) -> VectorX<T>;

    /// Conversion to dual space
    ///
    /// Simply convert all interior values of dtype to DD.
    fn dual_convert<TT: Numeric>(other: &Self::Alias<dtype>) -> Self::Alias<TT>;

    /// Dimension helper
    fn dim(&self) -> usize {
        Self::DIM
    }

    /// Adds value from the tangent space to the group element
    ///
    /// By default this uses the "right" version [^@solaMicroLieTheory2021]
    /// $$
    /// x \oplus \xi = x \cdot \exp(\xi)
    /// $$
    /// If the "left" feature is enabled, instead this turns to
    /// $$
    /// x \oplus \xi = \exp(\xi) \cdot x
    /// $$
    ///
    /// [^@solaMicroLieTheory2021]: Solà, Joan, et al. “A Micro Lie Theory for State Estimation in Robotics.” Arxiv:1812.01537, Dec. 2021
    fn oplus(&self, xi: VectorViewX<T>) -> Self {
        if cfg!(feature = "left") {
            self.oplus_left(xi)
        } else {
            self.oplus_right(xi)
        }
    }

    fn oplus_right(&self, xi: VectorViewX<T>) -> Self {
        self.compose(&Self::exp(xi))
    }

    fn oplus_left(&self, xi: VectorViewX<T>) -> Self {
        Self::exp(xi).compose(self)
    }

    /// Compares two group elements in the tangent space
    ///
    /// By default this uses the "right" version [^@solaMicroLieTheory2021]
    /// $$
    /// x \ominus y = \log(y^{-1} \cdot x)
    /// $$
    /// If the "left" feature is enabled, instead this turns to
    /// $$
    /// x \ominus y = \log(x \cdot y^{-1})
    /// $$
    ///
    /// [^@solaMicroLieTheory2021]: Solà, Joan, et al. “A Micro Lie Theory for State Estimation in Robotics.” Arxiv:1812.01537, Dec. 2021
    fn ominus(&self, y: &Self) -> VectorX<T> {
        if cfg!(feature = "left") {
            self.ominus_left(y)
        } else {
            self.ominus_right(y)
        }
    }

    fn ominus_right(&self, y: &Self) -> VectorX<T> {
        y.inverse().compose(self).log()
    }

    fn ominus_left(&self, y: &Self) -> VectorX<T> {
        self.compose(&y.inverse()).log()
    }

    /// Subtract out portion from other variable.
    ///
    /// This can be seen as a "tip-to-tail" computation. IE it computes the
    /// transformation between two poses. I like to think of it as "taking away"
    /// the portion subtracted out, for example given a chain of poses $a, b,
    /// c$, the following "removes" the portion from $a$ to $b$.
    ///
    /// $$
    /// {}_a T_c \boxminus {}_a T_b = ({}_a T_b)^{-1} {}_a T_c = {}_b T_c
    /// $$
    ///
    /// This operation is NOT effected by the left/right feature.
    fn minus(&self, other: &Self) -> Self {
        other.inverse().compose(self)
    }

    /// Setup group element correctly using the tangent space
    ///
    /// By default this uses the exponential map to propagate dual numbers to
    /// the variable to setup the jacobian properly. Can be hardcoded to avoid
    /// the repeated computation.
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

    /// Applies the tangent vector in dual space
    ///
    /// Takes the results from [dual_setup](Self::dual_setup) and applies the
    /// tangent vector using the right/left oplus operator.
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

/// The object safe version of [Variable].
///
/// This trait is used to allow for dynamic dispatch of noise models.
/// Implemented for all types that implement [Variable].
#[cfg_attr(feature = "serde", typetag::serde(tag = "tag"))]
pub trait VariableSafe: Debug + Display + Downcast {
    fn clone_box(&self) -> Box<dyn VariableSafe>;

    fn dim(&self) -> usize;

    fn oplus_mut(&mut self, delta: VectorViewX);
}

impl<
        #[cfg(not(feature = "serde"))] T: Variable + 'static,
        #[cfg(feature = "serde")] T: Variable + 'static + crate::serde::Tagged,
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

/// Umbrella trait for variables
///
/// This trait is 100% for convenience. It wraps all types that implements
/// [VariableSafe] and [Variable] (with proper aliases) into a single trait.
pub trait VariableUmbrella<T: Numeric = dtype>:
    VariableSafe + Variable<T, Alias<T> = Self>
{
}
impl<T: Numeric, V: VariableSafe + Variable<T, Alias<T> = V>> VariableUmbrella<T> for V {}

impl_downcast!(VariableSafe);

impl Clone for Box<dyn VariableSafe> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

use nalgebra as na;

/// Properties specific to matrix Lie groups
///
/// Many variables used in robotics state estimation are specific Lie Groups
/// that consist of matrix elements. We encapsulate a handful of their
/// properties here.
pub trait MatrixLieGroup<T: Numeric = dtype>: Variable<T>
where
    na::DefaultAllocator: na::allocator::Allocator<Self::TangentDim, Self::TangentDim>,
    na::DefaultAllocator: na::allocator::Allocator<Self::MatrixDim, Self::MatrixDim>,
    na::DefaultAllocator: na::allocator::Allocator<Self::VectorDim, Self::TangentDim>,
    na::DefaultAllocator: na::allocator::Allocator<Self::TangentDim, Const<1>>,
    na::DefaultAllocator: na::allocator::Allocator<Self::VectorDim, Const<1>>,
{
    /// Dimension of the tangent space
    type TangentDim: DimName;
    /// Dimension of the corresponding matrix representation
    type MatrixDim: DimName;
    /// Dimension of vectors that can be transformed
    type VectorDim: DimName;

    /// Adjoint operator
    fn adjoint(&self) -> MatrixDim<Self::TangentDim, Self::TangentDim, T>;

    /// Hat operator
    ///
    /// Converts a vector from $\xi \in \mathbb{R}^n$ to the Lie algebra
    /// $\xi^\wedge \in \mathfrak{g}$
    fn hat(
        xi: MatrixViewDim<'_, Self::TangentDim, Const<1>, T>,
    ) -> MatrixDim<Self::MatrixDim, Self::MatrixDim, T>;

    /// Vee operator
    ///
    /// Inverse of the hat operator. Converts a matrix from the Lie algebra
    /// $\xi^\wedge \in \mathfrak{g}$ to a vector $\xi \in \mathbb{R}^n$
    fn vee(
        xi: MatrixViewDim<'_, Self::MatrixDim, Self::MatrixDim, T>,
    ) -> MatrixDim<Self::TangentDim, Const<1>, T>;

    /// Hat operator for swapping
    ///
    /// This is our own version of the hat operator used for swapping with
    /// vectors to be rotated. For many common Lie groups, this encodes the
    /// following "swap"
    ///
    /// $$
    /// \xi^\wedge p = \text{hat\\_swap}(p) \xi
    /// $$
    ///
    ///
    /// For example, in SO(3) $\text{hat\\_swap}(p) = -p^\wedge$.
    fn hat_swap(
        xi: MatrixViewDim<'_, Self::VectorDim, Const<1>, T>,
    ) -> MatrixDim<Self::VectorDim, Self::TangentDim, T>;

    /// Transform a vector
    ///
    /// Transform/rotate a vector using the group element. In SO(3), this is
    /// rotation, in SE(3) this is a rigid body transformation.
    fn apply(
        &self,
        v: MatrixViewDim<'_, Self::VectorDim, Const<1>, T>,
    ) -> MatrixDim<Self::VectorDim, Const<1>, T>;

    /// Transform group element to a matrix
    fn to_matrix(&self) -> MatrixDim<Self::MatrixDim, Self::MatrixDim, T>;

    /// Create a group element from a matrix
    fn from_matrix(mat: MatrixViewDim<'_, Self::MatrixDim, Self::MatrixDim, T>) -> Self;
}
