//! Noise model representations
//!
//! Represent Gaussian noise models in a factor graph, specifically used when
//! constructing a [factor](crate::containers::Factor).

use std::fmt::{Debug, Display};

use crate::linalg::{DimName, MatrixX, VectorX};

/// The trait for a noise model.
pub trait NoiseModel: Debug + Display {
    /// The dimension of the noise model
    type Dim: DimName;

    fn dim(&self) -> usize {
        Self::Dim::USIZE
    }

    /// Whiten a vector
    fn whiten_vec(&self, v: VectorX) -> VectorX;

    /// Whiten a matrix
    fn whiten_mat(&self, m: MatrixX) -> MatrixX;
}

/// The object safe version of [NoiseModel].
///
/// This trait is used to allow for dynamic dispatch of noise models.
/// Implemented for all types that implement [NoiseModel].
#[cfg_attr(feature = "serde", typetag::serde(tag = "tag"))]
pub trait NoiseModelSafe: Debug + Display {
    fn dim(&self) -> usize;

    fn whiten_vec(&self, v: VectorX) -> VectorX;

    fn whiten_mat(&self, m: MatrixX) -> MatrixX;
}

impl<
        #[cfg(not(feature = "serde"))] T: NoiseModel,
        #[cfg(feature = "serde")] T: NoiseModel + crate::serde::Tagged,
    > NoiseModelSafe for T
{
    fn dim(&self) -> usize {
        NoiseModel::dim(self)
    }

    fn whiten_vec(&self, v: VectorX) -> VectorX {
        NoiseModel::whiten_vec(self, v)
    }

    fn whiten_mat(&self, m: MatrixX) -> MatrixX {
        NoiseModel::whiten_mat(self, m)
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

/// Register a type as a noise model for serde serialization.
#[macro_export]
macro_rules! tag_noise {
    ($($ty:ty),* $(,)?) => {$(
        $crate::register_typetag!($crate::noise::NoiseModelSafe, $ty);
    )*};
}

mod gaussian;
pub use gaussian::GaussianNoise;

mod unit;
pub use unit::UnitNoise;
