use crate::{
    dtype,
    linalg::{Matrix3, Numeric, Vector3},
    prelude::{MatrixLieGroup, Variable},
    variables::{ImuBias, SO3},
};

pub struct CombinedMeasurement<D: Numeric = dtype> {
    pub gyro: Vector3<D>,
    pub acc: Vector3<D>,
    pub dt: D,
}

pub struct PreintegratedCombinedMeasurements<D: Numeric = dtype> {
    bias: ImuBias<D>,
    theta: Vector3<D>,
    pos: Vector3<D>,
    vel: Vector3<D>,
}

impl PreintegratedCombinedMeasurements {
    pub fn integrate_measurement(&self, meas: CombinedMeasurement) {
        let rotation = SO3::exp(self.theta);
        // let H_inv = Matrix3::zeros();
        let norm_theta = self.theta.norm();

        // self.theta += (meas.gyro - self.bias.gyro) * meas.dt;
        // self.pos += self.vel * meas.dt + tmp * meas.dt / 2.0;
        // self.vel += tmp;
    }
}

pub struct CombinedImuResidual {}
