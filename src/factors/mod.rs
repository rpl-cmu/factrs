mod noise;
pub use noise::*;

use crate::make_enum_noise;
make_enum_noise!(
    NoiseModelEnum,
    GaussianNoise1,
    GaussianNoise2,
    GaussianNoise3,
    GaussianNoise4,
    GaussianNoise5,
    GaussianNoise6,
    GaussianNoise7,
    GaussianNoise8,
    GaussianNoise9,
    GaussianNoise10
);

mod residual;
use crate::make_enum_residual;
use crate::variables::{Vector3, SO3};
pub use residual::{BetweenResidual, PriorResidual};
// make_enum_residual!(
//     ResidualEnum,
//     BetweenResidual<Vector3>,
//     PriorResidual<Vector3>
// );

// macro_rules! test_macro {
//     ($name:ident $(< $($gen:ident)? >)? ) => {
//         println!("Hello, {}!", stringify!($name));
//     };
// }

// test_macro!(PriorResidual<SO3>);

pub enum ResidualEnum {
    BetweenResidual(BetweenResidual<SO3>),
    PriorResidual(PriorResidual<SO3>),
}

use crate::dtype;
use crate::traits::{Residual, Variable};
impl<V: crate::traits::Variable<crate::dtype>> crate::traits::Residual<V> for ResidualEnum
// where
//     V::Dual: std::convert::TryInto<<SO3 as Variable<dtype>>::Dual>,
{
    const DIM: usize = 0;

    fn residual(&self, v: &[V::Dual]) -> crate::linalg::VectorX<crate::traits::DualVec> {
        match self {
            ResidualEnum::BetweenResidual(x) => Residual::<V>::residual(x, v),
            ResidualEnum::PriorResidual(x) => Residual::<V>::residual(x, v),
        }
    }
}
