//! Struct & traits for implementing residuals
//!
//! Residuals are the main building block of a
//! [factor](crate::containers::Factor). To implement a custom residual, we
//! recommend implementing one of the numbered residual traits and then calling
//! the [impl_residual](crate::impl_residual) macro to implement [Residual].
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
