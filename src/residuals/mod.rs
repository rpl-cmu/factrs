mod traits;
pub use traits::{Residual, Residual1, Residual2, Residual3};

mod prior;
pub use prior::PriorResidual;

mod between;
pub use between::BetweenResidual;

mod macros;
use crate::make_enum_residual;
use crate::variables::{VariableEnum, Vector3, SE3, SO3};
// TODO: Add everything to this
make_enum_residual!(
    ResidualEnum,
    VariableEnum,
    BetweenResidual<Vector3>,
    PriorResidual<Vector3>,
    PriorResidual<SO3>,
    PriorResidual<SE3>
);
