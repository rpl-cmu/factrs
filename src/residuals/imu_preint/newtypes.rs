use crate::{
    dtype,
    linalg::{Numeric, Vector3},
    variables::{ImuBias, SO3},
};

/// Raw gyro measurement
///
/// This is a newtype for the gyro measurement ensure that the accel and gyro
/// aren't mixed up.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct Gyro<T: Numeric = dtype>(pub Vector3<T>);

impl<T: Numeric> Gyro<T> {
    /// Remove the bias from the gyro measurement
    pub fn remove_bias(&self, bias: &ImuBias<T>) -> GyroUnbiased<T> {
        GyroUnbiased(self.0 - bias.gyro())
    }

    pub fn zeros() -> Self {
        Gyro(Vector3::zeros())
    }

    pub fn new(x: T, y: T, z: T) -> Self {
        Gyro(Vector3::new(x, y, z))
    }
}

/// Gyro measurement with bias removed
pub struct GyroUnbiased<T: Numeric = dtype>(pub Vector3<T>);

/// Raw accel measurement newtype
///
/// This is a newtype for the accel measurement ensure that the accel and gyro
/// aren't mixed up.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct Accel<T: Numeric = dtype>(pub Vector3<T>);

impl<T: Numeric> Accel<T> {
    /// Remove the bias from the accel measurement
    pub fn remove_bias(&self, bias: &ImuBias<T>) -> AccelUnbiased<T> {
        AccelUnbiased(self.0 - bias.accel())
    }

    pub fn zeros() -> Self {
        Accel(Vector3::zeros())
    }

    pub fn new(x: T, y: T, z: T) -> Self {
        Accel(Vector3::new(x, y, z))
    }
}

/// Accel measurement with bias removed
pub struct AccelUnbiased<T: Numeric = dtype>(pub Vector3<T>);

/// Gravity vector
///
/// This is a newtype for the gravity vector to ensure that it is not mixed up
/// with other vectors and to provide some convenience methods.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct Gravity<T: Numeric = dtype>(pub Vector3<T>);

impl<T: Numeric> Gravity<T> {
    /// Helper to get the gravity vector pointing up, i.e. [0, 0, 9.81]
    pub fn up() -> Self {
        Gravity(Vector3::new(T::from(0.0), T::from(0.0), T::from(9.81)))
    }

    /// Helper to get the gravity vector pointing down, i.e. [0, 0, -9.81]
    pub fn down() -> Self {
        Gravity(Vector3::new(T::from(0.0), T::from(0.0), T::from(-9.81)))
    }
}

/// Struct to hold an Imu state
///
/// Specifically holds an Imu state to which an ImuDelta can be applied
pub struct ImuState<T: Numeric = dtype> {
    pub r: SO3<T>,
    pub v: Vector3<T>,
    pub p: Vector3<T>,
    pub bias: ImuBias<T>,
}
