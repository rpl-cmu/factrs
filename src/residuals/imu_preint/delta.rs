use core::fmt;

use super::{Accel, Gravity, Gyro, ImuCovariance, ImuState};
use crate::{
    dtype,
    linalg::{Matrix, Matrix3, Numeric, Vector3},
    traits::{DualConvert, Variable},
    variables::{ImuBias, MatrixLieGroup, SO3},
};

/// Struct to hold the preintegrated Imu delta
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub(crate) struct ImuDelta<D: Numeric = dtype> {
    dt: D,
    xi_theta: Vector3<D>,
    xi_vel: Vector3<D>,
    xi_pos: Vector3<D>,
    bias_init: ImuBias<D>,
    h_bias_accel: Matrix<9, 3, D>,
    h_bias_gyro: Matrix<9, 3, D>,
    gravity: Gravity<D>,
}

impl<D: Numeric> ImuDelta<D> {
    pub(crate) fn new(gravity: Gravity<D>, bias_init: ImuBias<D>) -> Self {
        Self {
            dt: D::from(0.0),
            xi_theta: Vector3::zeros(),
            xi_vel: Vector3::zeros(),
            xi_pos: Vector3::zeros(),
            bias_init,
            h_bias_accel: Matrix::zeros(),
            h_bias_gyro: Matrix::zeros(),
            gravity,
        }
    }

    pub(crate) fn bias_init(&self) -> &ImuBias<D> {
        &self.bias_init
    }

    pub(crate) fn first_order_update(&self, bias_diff: &ImuBias<D>, idx: usize) -> Vector3<D> {
        self.h_bias_accel.fixed_rows(idx) * bias_diff.accel()
            + self.h_bias_gyro.fixed_rows(idx) * bias_diff.gyro()
    }

    pub(crate) fn predict(&self, x1: &ImuState<D>) -> ImuState<D> {
        let ImuState { r, v, p, bias } = x1;

        // Compute first-order updates to preint with bias
        let bias_diff = bias - &self.bias_init;
        let xi_theta = self.xi_theta + self.first_order_update(&bias_diff, 0);
        let xi_v = self.xi_vel + self.first_order_update(&bias_diff, 3);
        let xi_p = self.xi_pos + self.first_order_update(&bias_diff, 6);

        // Estimate x2
        // R2_meas = R1 * exp(xi_theta)
        let r2_meas = r.compose(&SO3::exp(xi_theta.as_view()));

        // v2_meas = v + g * dt + R1 * xi_v
        let v2_meas = v + self.gravity.0 * self.dt + r.apply(xi_v.as_view());

        // p2_meas = p1 + v * dt + 0.5 * g * dt^2 + R1 * xi_p
        let p2_meas = p
            + v * self.dt
            + self.gravity.0 * self.dt * self.dt * D::from(0.5)
            + r.apply(xi_p.as_view());

        let b2_meas = bias.clone();

        ImuState {
            r: r2_meas,
            v: v2_meas,
            p: p2_meas,
            bias: b2_meas,
        }
    }
}

// None of these methods should need to be dualized / "backpropagated"
// Can just assume default dtype
impl ImuDelta {
    #[allow(non_snake_case)]
    pub(crate) fn propagate_H(&mut self, A: &Matrix<15, 15>) {
        let Amini = A.fixed_view::<9, 9>(0, 0);
        // This should come from B, turns out it's identical as A
        let Bgyro = A.fixed_view::<9, 3>(0, 9);
        let Baccel = A.fixed_view::<9, 3>(0, 12);
        self.h_bias_gyro = Amini * self.h_bias_gyro + Bgyro;
        self.h_bias_accel = Amini * self.h_bias_accel + Baccel;
    }

    #[allow(non_snake_case)]
    pub(crate) fn integrate(&mut self, gyro: &Gyro, accel: &Accel, dt: dtype) {
        self.dt += dt;
        let accel_world = SO3::exp(self.xi_theta.as_view()).apply(accel.0.as_view());
        let H = SO3::dexp(self.xi_theta.as_view());
        let Hinv = H.try_inverse().expect("Failed to invert H(theta)");

        self.xi_theta += Hinv * gyro.0 * dt;
        self.xi_vel += accel_world * dt;
        self.xi_pos += self.xi_vel * dt + accel_world * (dt * dt * 0.5);
    }

    #[allow(non_snake_case)]
    pub(crate) fn A(&self, gyro: &Gyro, accel: &Accel, dt: dtype) -> Matrix<15, 15> {
        let H = SO3::dexp(self.xi_theta.as_view());
        let Hinv = H.try_inverse().expect("Failed to invert H(theta)");
        let R = SO3::exp(self.xi_theta.as_view()).to_matrix();

        let mut A = Matrix::<15, 15>::identity();

        // First column (wrt theta)
        let M = Matrix3::identity() - SO3::hat(gyro.0.as_view()) * dt / 2.0;
        A.fixed_view_mut::<3, 3>(0, 0).copy_from(&M);
        let mut M = -R * SO3::hat(accel.0.as_view()) * H * dt;
        A.fixed_view_mut::<3, 3>(3, 0).copy_from(&M);
        M *= dt / 2.0;
        A.fixed_view_mut::<3, 3>(6, 0).copy_from(&M);

        // Second column (wrt vel)
        let M = Matrix3::identity() * dt;
        A.fixed_view_mut::<3, 3>(0, 6).copy_from(&M);

        // Third column (wrt pos)

        // Fourth column (wrt gyro bias)
        let M = -Hinv * dt;
        A.fixed_view_mut::<3, 3>(0, 9).copy_from(&M);

        // Fifth column (wrt accel bias)
        let mut M = -R * dt;
        A.fixed_view_mut::<3, 3>(6, 9).copy_from(&M);
        M *= dt / 2.0;
        A.fixed_view_mut::<3, 3>(3, 9).copy_from(&M);

        A
    }

    #[allow(non_snake_case)]
    pub(crate) fn B_Q_BT(&self, p: &ImuCovariance, dt: dtype) -> Matrix<15, 15> {
        let H = SO3::dexp(self.xi_theta.as_view());
        let Hinv = H.try_inverse().expect("Failed to invert H(theta)");
        let R = SO3::exp(self.xi_theta.as_view()).to_matrix();

        // Construct all partials
        let H_theta_w = Hinv * dt;
        let H_theta_winit = -Hinv * self.dt;

        let H_v_a = R * dt;
        let H_v_ainit = -R * self.dt;

        let H_p_a = H_v_a * dt / 2.0;
        let H_p_int: Matrix3<dtype> = Matrix3::identity();
        let H_p_ainit = H_v_ainit * self.dt / 2.0;

        // Copy them into place
        let mut B = Matrix::<15, 15>::zeros();
        // First column (wrt theta)
        let M = H_theta_w * p.cov_gyro_bias * H_theta_w.transpose()
            + H_theta_winit * p.cov_winit * H_theta_winit.transpose();
        B.fixed_view_mut::<3, 3>(0, 0).copy_from(&M);

        // Second column (wrt vel)
        let M = H_v_a * p.cov_accel * H_v_a.transpose()
            + H_v_ainit * p.cov_ainit * H_v_ainit.transpose();
        B.fixed_view_mut::<3, 3>(3, 3).copy_from(&M);
        let M = H_p_a * p.cov_accel * H_v_a.transpose()
            + H_p_ainit * p.cov_ainit * H_v_ainit.transpose();
        B.fixed_view_mut::<3, 3>(6, 3).copy_from(&M);

        // Third column (wrt pos)
        let M = H_v_a * p.cov_accel * H_p_a.transpose()
            + H_v_ainit * p.cov_ainit * H_p_ainit.transpose();
        B.fixed_view_mut::<3, 3>(3, 6).copy_from(&M);
        let M = H_p_a * p.cov_accel * H_p_a.transpose()
            + H_p_int * p.cov_integration * H_p_int.transpose()
            + H_p_ainit * p.cov_ainit * H_p_ainit.transpose();
        B.fixed_view_mut::<3, 3>(6, 6).copy_from(&M);

        // Fourth column (wrt gyro bias)
        B.fixed_view_mut::<3, 3>(9, 9).copy_from(&p.cov_gyro_bias);

        // Fifth column (wrt accel bias)
        B.fixed_view_mut::<3, 3>(12, 12)
            .copy_from(&p.cov_accel_bias);

        B
    }
}

impl<D: Numeric> DualConvert for ImuDelta<D> {
    type Alias<DD: Numeric> = ImuDelta<DD>;
    fn dual_convert<DD: Numeric>(other: &Self::Alias<dtype>) -> Self::Alias<DD> {
        ImuDelta {
            dt: other.dt.into(),
            xi_theta: Vector3::<D>::dual_convert(&other.xi_theta),
            xi_vel: Vector3::<D>::dual_convert(&other.xi_vel),
            xi_pos: Vector3::<D>::dual_convert(&other.xi_pos),
            bias_init: ImuBias::<D>::dual_convert(&other.bias_init),
            h_bias_accel: Matrix::<9, 3, D>::dual_convert(&other.h_bias_accel),
            h_bias_gyro: Matrix::<9, 3, D>::dual_convert(&other.h_bias_gyro),
            gravity: Gravity(Vector3::<D>::dual_convert(&other.gravity.0)),
        }
    }
}

impl<D: Numeric> fmt::Display for ImuDelta<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "ImuDelta(dt: {}, theta: {}, v: {}, p: {})",
            self.dt, self.xi_theta, self.xi_vel, self.xi_pos
        )
    }
}
