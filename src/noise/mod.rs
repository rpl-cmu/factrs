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

#[macro_export]
macro_rules! impl_safe_noise {
    ($($var:ident < $num:literal > ),* $(,)?) => {
        use paste::paste;
        paste!{
            $(
                type [<$var $num>] = $var< $num >;
                #[cfg_attr(feature = "serde", typetag::serde)]
                impl $crate::noise::NoiseModelSafe for [<$var $num>]{
                    fn dim(&self) -> usize {
                        $crate::noise::NoiseModel::dim(self)
                    }

                    fn whiten_vec(&self, v: VectorX) -> VectorX {
                        $crate::noise::NoiseModel::whiten_vec(self, v)
                    }

                    fn whiten_mat(&self, m: MatrixX) -> MatrixX {
                        $crate::noise::NoiseModel::whiten_mat(self, m)
                    }
                }
            )*
        }
    };
}

mod gaussian;
pub use gaussian::GaussianNoise;

mod unit;
pub use unit::UnitNoise;
