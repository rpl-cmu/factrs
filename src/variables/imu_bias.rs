use crate::{dtype, linalg::Numeric, linalg::Vector3};

pub struct ImuBias<D: Numeric = dtype> {
    pub gyro: Vector3<D>,
    pub acc: Vector3<D>,
}
