#[allow(non_camel_case_types)]
pub type dtype = f64;

#[cfg(feature = "f32")]
pub type dtype = f32;

pub mod bundle;
pub mod containers;
pub mod factors;
pub mod linalg;
pub mod noise;
pub mod residuals;
pub mod robust;
pub mod traits;
pub mod utils;
pub mod variables;
