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
use crate::variables::{VariableEnum, Vector3, SE3, SO3};
pub use residual::{BetweenResidual, PriorResidual};

make_enum_residual!(
    ResidualEnum,
    VariableEnum,
    BetweenResidual<Vector3>,
    PriorResidual<Vector3>,
    PriorResidual<SO3>,
    PriorResidual<SE3>
);
