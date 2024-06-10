#[allow(non_camel_case_types)]
pub type dtype = f64;

#[cfg(feature = "f32")]
pub type dtype = f32;

pub mod factors;
pub mod linalg;
pub mod macros;
pub mod residuals;
pub mod traits;
pub mod variables;

// struct DefaultBundle;

// impl traits::Bundle for DefaultBundle {
//     type Key = variables::Symbol;
//     type Variable = variables::VariableEnum;
//     type Robust = (); // TODO
//     type Noise = factors::NoiseModelEnum;
//     type Residual = factors::ResidualEnum;
// }
