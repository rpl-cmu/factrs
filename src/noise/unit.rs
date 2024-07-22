use super::NoiseModel;
use crate::{
    linalg::{Const, MatrixX, VectorX},
    tag_noise,
};

tag_noise!(
    UnitNoise<1>,
    UnitNoise<2>,
    UnitNoise<3>,
    UnitNoise<4>,
    UnitNoise<5>,
    UnitNoise<6>,
    UnitNoise<7>,
    UnitNoise<8>,
    UnitNoise<9>,
    UnitNoise<10>,
    UnitNoise<11>,
    UnitNoise<12>,
);

/// A unit noise model.
///
/// Represents a noise model that does not modify the input, or equal weighting
/// in a [factor](crate::factor::Factor).
#[derive(Clone, Debug, derive_more::Display)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct UnitNoise<const N: usize>;

impl<const N: usize> NoiseModel for UnitNoise<N> {
    type Dim = Const<N>;

    fn whiten_vec(&self, v: VectorX) -> VectorX {
        v
    }

    fn whiten_mat(&self, m: MatrixX) -> MatrixX {
        m
    }
}
