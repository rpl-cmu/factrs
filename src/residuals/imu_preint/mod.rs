#![doc = include_str!("README.md")]

mod newtypes;
pub(crate) use newtypes::ImuState;
pub use newtypes::{Accel, AccelUnbiased, Gravity, Gyro, GyroUnbiased};

mod delta;

mod residual;
pub use residual::{ImuCovariance, ImuPreintegrator};
