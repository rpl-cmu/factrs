mod noise;
pub use noise::*;

use crate::make_enum_noise;
use crate::traits::NoiseModel;
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
