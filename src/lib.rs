#[cfg(not(feature = "f32"))]
#[allow(non_camel_case_types)]
pub type dtype = f64;

#[cfg(feature = "f32")]
#[allow(non_camel_case_types)]
pub type dtype = f32;

pub mod bundle;
pub mod containers;
pub mod factors;
pub mod linalg;
pub mod linear;
pub mod noise;
pub mod optimizers;
pub mod residuals;
pub mod robust;
pub mod utils;
pub mod variables;
