mod noise;
pub use noise::GaussianNoise;

mod residual;
use crate::variables::{VariableEnum, Vector3, SE3, SO3};
use crate::{make_enum_residual, make_enum_robust};
pub use residual::{BetweenResidual, PriorResidual};
// TODO: Add everything to this
make_enum_residual!(
    ResidualEnum,
    VariableEnum,
    BetweenResidual<Vector3>,
    PriorResidual<Vector3>,
    PriorResidual<SO3>,
    PriorResidual<SE3>
);

mod robust;
pub use robust::*;
make_enum_robust!(
    RobustEnum,
    L2,
    L1,
    Huber,
    Fair,
    Cauchy,
    GemanMcClure,
    Welsch,
    Tukey
);

mod factor;
pub use factor::Factor;
