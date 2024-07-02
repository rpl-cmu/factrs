use std::fmt::{Debug, Display};

use crate::linalg::{DimName, MatrixViewX, MatrixX, VectorViewX, VectorX};
pub trait NoiseModel: Debug + Display {
    type Dim: DimName;

    fn dim(&self) -> usize {
        Self::Dim::USIZE
    }

    fn whiten_vec(&self, v: VectorViewX) -> VectorX;

    fn whiten_mat(&self, m: MatrixViewX) -> MatrixX;
}

#[cfg_attr(feature = "serde", typetag::serde(tag = "type"))]
pub trait NoiseModelSafe: Debug + Display {
    fn dim(&self) -> usize;

    fn whiten_vec(&self, v: VectorViewX) -> VectorX;

    fn whiten_mat(&self, m: MatrixViewX) -> MatrixX;
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

                    fn whiten_vec(&self, v: VectorViewX) -> VectorX {
                        $crate::noise::NoiseModel::whiten_vec(self, v)
                    }

                    fn whiten_mat(&self, m: MatrixViewX) -> MatrixX {
                        $crate::noise::NoiseModel::whiten_mat(self, m)
                    }
                }
            )*
        }
    };
}

mod gaussian;
pub use gaussian::GaussianNoise;
