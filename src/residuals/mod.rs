mod traits;
pub use traits::{
    Residual,
    Residual1,
    Residual2,
    Residual3,
    Residual4,
    Residual5,
    Residual6,
    ResidualSafe,
};

mod prior;
pub use prior::PriorResidual;

mod between;
pub use between::BetweenResidual;

mod macros;
