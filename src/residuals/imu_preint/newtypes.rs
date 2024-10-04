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
pub struct Gyro<D: Numeric = dtype>(pub Vector3<D>);

impl<D: Numeric> Gyro<D> {
    /// Remove the bias from the gyro measurement
    pub fn remove_bias(&self, bias: &ImuBias<D>) -> GyroUnbiased<D> {
        GyroUnbiased(self.0 - bias.gyro())
    }

    pub fn zeros() -> Self {
        Gyro(Vector3::zeros())
    }

    pub fn new(x: D, y: D, z: D) -> Self {
        Gyro(Vector3::new(x, y, z))
    }
}

/// Gyro measurement with bias removed
pub struct GyroUnbiased<D: Numeric = dtype>(pub Vector3<D>);

/// Raw accel measurement newtype
///
/// This is a newtype for the accel measurement ensure that the accel and gyro
/// aren't mixed up.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct Accel<D: Numeric = dtype>(pub Vector3<D>);

impl<D: Numeric> Accel<D> {
    /// Remove the bias from the accel measurement
    pub fn remove_bias(&self, bias: &ImuBias<D>) -> AccelUnbiased<D> {
        AccelUnbiased(self.0 - bias.accel())
    }

    pub fn zeros() -> Self {
        Accel(Vector3::zeros())
    }

    pub fn new(x: D, y: D, z: D) -> Self {
        Accel(Vector3::new(x, y, z))
    }
}

/// Accel measurement with bias removed
pub struct AccelUnbiased<D: Numeric = dtype>(pub Vector3<D>);

/// Gravity vector
///
/// This is a newtype for the gravity vector to ensure that it is not mixed up
/// with other vectors and to provide some convenience methods.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct Gravity<D: Numeric = dtype>(pub Vector3<D>);

impl<D: Numeric> Gravity<D> {
    /// Helper to get the gravity vector pointing up, i.e. [0, 0, 9.81]
    pub fn up() -> Self {
        Gravity(Vector3::new(D::from(0.0), D::from(0.0), D::from(9.81)))
    }

    /// Helper to get the gravity vector pointing down, i.e. [0, 0, -9.81]
    pub fn down() -> Self {
        Gravity(Vector3::new(D::from(0.0), D::from(0.0), D::from(-9.81)))
    }
}

/// Struct to hold an Imu state
///
/// Specifically holds an Imu state to which an ImuDelta can be applied
pub struct ImuState<D: Numeric = dtype> {
    pub r: SO3<D>,
    pub v: Vector3<D>,
    pub p: Vector3<D>,
    pub bias: ImuBias<D>,
}
