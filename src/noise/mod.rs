use std::fmt::{Debug, Display};

use crate::linalg::{DimName, MatrixX, VectorX};

pub trait NoiseModel: Debug + Display {
    type Dim: DimName;

    fn dim(&self) -> usize {
        Self::Dim::USIZE
    }

    fn whiten_vec(&self, v: VectorX) -> VectorX;

    fn whiten_mat(&self, m: MatrixX) -> MatrixX;
}

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

#[macro_export]
macro_rules! register_noise {
    ($($ty:ty),* $(,)?) => {$(
        $crate::register_typetag!($crate::noise::NoiseModelSafe, $ty);
    )*};
}

mod gaussian;
pub use gaussian::GaussianNoise;

mod unit;
pub use unit::UnitNoise;
