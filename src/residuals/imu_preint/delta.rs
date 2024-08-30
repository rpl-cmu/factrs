use core::fmt;

use super::{AccelUnbiased, Gravity, GyroUnbiased, ImuCovariance, ImuState};
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

    // TODO: Should this also propagate the H matrix?
    // We'll have to construct A if we want to do that...
    #[allow(non_snake_case)]
    pub(crate) fn integrate(&mut self, gyro: &GyroUnbiased<D>, accel: &AccelUnbiased<D>, dt: D) {
        self.dt += dt;
        let accel_world = SO3::exp(self.xi_theta.as_view()).apply(accel.0.as_view());
        let H = SO3::dexp(self.xi_theta.as_view());
        let Hinv = H.try_inverse().expect("Failed to invert H(theta)");

        // Integrate (make sure integration occurs in the correct order)
        self.xi_pos += self.xi_vel * dt + accel_world * (dt * dt * 0.5);
        self.xi_vel += accel_world * dt;
        self.xi_theta += Hinv * gyro.0 * dt;
    }

    #[allow(non_snake_case)]
    pub(crate) fn propagate_H(&mut self, A: &Matrix<15, 15, D>) {
        let Amini = A.fixed_view::<9, 9>(0, 0);
        // This should come from B, turns out it's identical as A
        let Bgyro = A.fixed_view::<9, 3>(0, 9);
        let Baccel = A.fixed_view::<9, 3>(0, 12);
        self.h_bias_gyro = Amini * self.h_bias_gyro + Bgyro;
        self.h_bias_accel = Amini * self.h_bias_accel + Baccel;
    }

    #[allow(non_snake_case)]
    pub(crate) fn A(
        &self,
        gyro: &GyroUnbiased<D>,
        accel: &AccelUnbiased<D>,
        dt: D,
    ) -> Matrix<15, 15, D> {
        let H = SO3::dexp(self.xi_theta.as_view());
        let Hinv = H.try_inverse().expect("Failed to invert H(theta)");
        let R: nalgebra::Matrix<
            D,
            nalgebra::Const<3>,
            nalgebra::Const<3>,
            nalgebra::ArrayStorage<D, 3, 3>,
        > = SO3::exp(self.xi_theta.as_view()).to_matrix();

        let mut A = Matrix::<15, 15, D>::identity();

        // First column (wrt theta)
        let M = Matrix3::<D>::identity() - SO3::hat(gyro.0.as_view()) * dt / D::from(2.0);
        A.fixed_view_mut::<3, 3>(0, 0).copy_from(&M);
        let mut M = -R * SO3::hat(accel.0.as_view()) * H * dt;
        A.fixed_view_mut::<3, 3>(3, 0).copy_from(&M);
        M *= dt / 2.0;
        A.fixed_view_mut::<3, 3>(6, 0).copy_from(&M);

        // Second column (wrt vel)
        let M = Matrix3::<D>::identity() * dt;
        A.fixed_view_mut::<3, 3>(6, 3).copy_from(&M);

        // Third column (wrt pos)

        // Fourth column (wrt gyro bias)
        let M = -Hinv * dt;
        A.fixed_view_mut::<3, 3>(0, 9).copy_from(&M);

        // Fifth column (wrt accel bias)
        let mut M = -R * dt;
        A.fixed_view_mut::<3, 3>(3, 12).copy_from(&M);
        M *= dt / 2.0;
        A.fixed_view_mut::<3, 3>(6, 12).copy_from(&M);

        A
    }
}

// This functions will never need to be backpropped through
impl ImuDelta {
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

#[cfg(test)]
mod test {
    use matrixcompare::{assert_matrix_eq, assert_scalar_eq};
    use nalgebra::Const;

    use super::*;
    use crate::{
        linalg::{DualVector, ForwardProp, VectorX},
        residuals::{Accel, Gyro},
        traits::*,
    };

    // Helper function to integrate a constant gyro / accel for a given amount of
    // time
    fn integrate<D: Numeric>(
        gyro: &Gyro<D>,
        accel: &Accel<D>,
        bias: &ImuBias<D>,
        n: i32,
        t: dtype,
    ) -> ImuDelta<D> {
        let mut delta = ImuDelta::new(Gravity::up(), bias.clone());
        // Remove the bias
        let accel = accel.remove_bias(bias);
        let gyro = gyro.remove_bias(bias);
        let dt = D::from(t / n as dtype);

        for _ in 0..n {
            // propagate H
            let a = delta.A(&gyro, &accel, dt);
            delta.propagate_H(&a);
            // Integrate values
            delta.integrate(&gyro, &accel, dt);
        }

        delta
    }

    // Test contast acceleration
    #[test]
    fn integrate_accel() {
        let a = 5.0;
        let t = 3.0;
        let n = 100;

        let accel = Accel(Vector3::new(a, 0.0, 0.0));
        let gyro = Gyro(Vector3::zeros());

        let delta = integrate(&gyro, &accel, &ImuBias::identity(), n, t);

        println!("Delta: {}", delta);
        assert_scalar_eq!(delta.dt, t, comp = abs, tol = 1e-5);
        assert_matrix_eq!(delta.xi_vel, accel.0 * t, comp = abs, tol = 1e-5);
        assert_matrix_eq!(delta.xi_pos, accel.0 * t * t / 2.0, comp = abs, tol = 1e-5);
    }

    // Test constant angular velocity
    #[test]
    fn integrate_gyro() {
        let a = 5.0;
        let t = 3.0;
        let n = 100;

        let accel = Accel(Vector3::zeros());
        let gyro = Gyro(Vector3::new(0.0, 0.0, a));

        let delta = integrate(&gyro, &accel, &ImuBias::identity(), n, t);

        println!("Delta: {}", delta);
        assert_scalar_eq!(delta.dt, t, comp = abs, tol = 1e-5);
        assert_matrix_eq!(delta.xi_theta, gyro.0 * t, comp = abs, tol = 1e-5);
    }

    #[test]
    #[allow(non_snake_case)]
    fn propagate_h_accel() {
        let t = 1.0;
        let n = 2;
        let accel = Accel::new(1.0, 2.0, 3.0);
        let gyro = Gyro::new(0.1, 0.2, 0.3);
        let bias = ImuBias::new(Gyro::new(1.0, 2.0, 3.0), Accel::new(0.1, 0.2, 0.3));

        // Compute the H matrix
        let delta = integrate(&gyro, &accel, &bias, n, t);
        let H_accel_got = delta.h_bias_accel;
        let H_gyro_got = delta.h_bias_gyro;

        // Compute the H matrix via forward prop
        let integrate_diff = |bias: ImuBias<DualVector<Const<6>>>| {
            let accel = Accel(accel.0.map(|a| a.into()));
            let gyro = Gyro(gyro.0.map(|g| g.into()));
            let delta = integrate(&gyro, &accel, &bias, n, t);
            let mut preint = VectorX::zeros(9);
            preint.fixed_rows_mut::<3>(0).copy_from(&delta.xi_theta);
            preint.fixed_rows_mut::<3>(3).copy_from(&delta.xi_vel);
            preint.fixed_rows_mut::<3>(6).copy_from(&delta.xi_pos);
            preint
        };
        let H_exp = ForwardProp::<Const<6>>::jacobian_1(integrate_diff, &bias).diff;
        let H_gyro_exp = H_exp.fixed_view::<9, 3>(0, 0);
        let H_accel_exp = H_exp.fixed_view::<9, 3>(0, 3);

        println!("H_accel_got: {:.4}", H_accel_got);
        println!("H_accel_exp: {:.4}", H_accel_exp);
        assert_matrix_eq!(H_accel_got, H_accel_exp, comp = abs, tol = 1e-5);

        // TODO: Gyro still failing
        println!("H_gyro_got: {:.4}", H_gyro_got);
        println!("H_gyro_exp: {:.4}", H_gyro_exp);
        assert_matrix_eq!(H_gyro_got, H_gyro_exp, comp = abs, tol = 1e-5);
    }
}
