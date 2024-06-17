mod traits;
pub use traits::{Residual, Residual1, Residual2, Residual3, Residual4, Residual5, Residual6};

mod prior;
pub use prior::PriorResidual;

mod between;
pub use between::BetweenResidual;

mod macros;
use crate::make_enum_residual;
use crate::variables::*;
// TODO: Add everything to this
make_enum_residual!(
    ResidualEnum,
    VariableEnum,
    BetweenResidual<Vector1>,
    BetweenResidual<Vector2>,
    BetweenResidual<Vector3>,
    BetweenResidual<Vector4>,
    BetweenResidual<Vector5>,
    BetweenResidual<Vector6>,
    BetweenResidual<SO3>,
    BetweenResidual<SE3>,
    PriorResidual<Vector1>,
    PriorResidual<Vector2>,
    PriorResidual<Vector3>,
    PriorResidual<Vector4>,
    PriorResidual<Vector5>,
    PriorResidual<Vector6>,
    PriorResidual<SO3>,
    PriorResidual<SE3>
);
