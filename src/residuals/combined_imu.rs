use crate::{
    dtype,
    linalg::{Vector3,Matrix3},
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

impl<D: Numeric = dtype> PreintegratedCombinedMeasurements<D> {
    pub fn integrate_measurement(&self, meas: CombinedMeasurement<D>) {
        tmp = SO3::exp(self.theta).apply(meas.acc) * meas.dt;
        H = Matrix3<D>::zero();
        self.theta += (meas.gyro - self.bias.gyro) * meas.dt;
        self.pos += self.vel * dt + tmp * meas.dt / 2.0;
        self.vel += tmp;
    }
}

pub struct CombinedImuResidual {}
