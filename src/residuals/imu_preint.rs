use core::fmt;
use std::ops;

use nalgebra::Const;

use super::Residual6;
use crate::{
    containers::{Factor, FactorBuilder, TypedSymbol},
    dtype,
    impl_residual,
    linalg::{ForwardProp, Matrix, Matrix3, Numeric, Vector3, VectorX},
    noise::GaussianNoise,
    tag_residual,
    traits::*,
    variables::{ImuBias, MatrixLieGroup, VectorVar3, SE3, SO3},
};
// ------------------- Objects that perform most of work ------------------- //
/// Struct to hold an Imu state
///
/// Specifically holds an Imu state to which an ImuDelta can be applied
pub struct ImuState<D: Numeric = dtype> {
    r: SO3<D>,
    v: Vector3<D>,
    p: Vector3<D>,
    bias: ImuBias<D>,
}

impl<D: Numeric> ImuState<D> {
    /// Propagate the state forward by the given delta
    ///
    /// This function takes an ImuDelta and applies it to the current state.
    /// Identical to calling [ImuDelta::predict] with the current state.
    pub fn propagate(&self, delta: ImuDelta<D>) -> ImuState<D> {
        delta.predict(self)
    }
}

/// Struct to hold the preintegrated Imu delta
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct ImuDelta<D: Numeric = dtype> {
    dt: D,
    xi_theta: Vector3<D>,
    xi_vel: Vector3<D>,
    xi_pos: Vector3<D>,
    bias_init: ImuBias<D>,
    h_bias_accel: Matrix<9, 3, D>,
    h_bias_gyro: Matrix<9, 3, D>,
    gravity: Vector3<D>,
}

impl<D: Numeric> ImuDelta<D> {
    fn new(gravity: Vector3<D>, bias_init: ImuBias<D>) -> Self {
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

    fn remove_bias(&self, imu: &ImuMeasurement<D>) -> ImuMeasurement<D> {
        imu - &self.bias_init
    }

    fn first_order_update(&self, bias_diff: &ImuBias<D>, idx: usize) -> Vector3<D> {
        self.h_bias_accel.fixed_rows(idx) * bias_diff.accel()
            + self.h_bias_gyro.fixed_rows(idx) * bias_diff.gyro()
    }

    pub fn predict(&self, x1: &ImuState<D>) -> ImuState<D> {
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
        let v2_meas = v + self.gravity * self.dt + r.apply(xi_v.as_view());
        let v2_meas = v2_meas.into();
        // p2_meas = p1 + v * dt + 0.5 * g * dt^2 + R1 * xi_p
        let p2_meas = p
            + v * self.dt
            + self.gravity * self.dt * self.dt * D::from(0.5)
            + r.apply(xi_p.as_view());
        let p2_meas = p2_meas.into();
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
    pub fn integrate(&mut self, imu: &ImuMeasurement) {
        self.dt += imu.dt;
        let accel_world = SO3::exp(self.xi_theta.as_view()).apply(imu.accel.as_view());
        let H = SO3::dexp(self.xi_theta.as_view());
        let Hinv = H.try_inverse().expect("Failed to invert H(theta)");

        self.xi_theta += Hinv * imu.gyro * imu.dt;
        self.xi_vel += accel_world * imu.dt;
        self.xi_pos += self.xi_vel * imu.dt + accel_world * (imu.dt * imu.dt * 0.5);
    }

    #[allow(non_snake_case)]
    pub fn A(&self, imu: &ImuMeasurement) -> Matrix<15, 15> {
        let H = SO3::dexp(self.xi_theta.as_view());
        let Hinv = H.try_inverse().expect("Failed to invert H(theta)");
        let R = SO3::exp(self.xi_theta.as_view()).to_matrix();

        let mut A = Matrix::<15, 15>::identity();

        // First column (wrt theta)
        let M = Matrix3::identity() - SO3::hat(imu.gyro.as_view()) * imu.dt / 2.0;
        A.fixed_view_mut::<3, 3>(0, 0).copy_from(&M);
        let mut M = -R * SO3::hat(imu.accel.as_view()) * H * imu.dt;
        A.fixed_view_mut::<3, 3>(3, 0).copy_from(&M);
        M *= imu.dt / 2.0;
        A.fixed_view_mut::<3, 3>(6, 0).copy_from(&M);

        // Second column (wrt vel)
        let M = Matrix3::identity() * imu.dt;
        A.fixed_view_mut::<3, 3>(0, 6).copy_from(&M);

        // Third column (wrt pos)

        // Fourth column (wrt gyro bias)
        let M = -Hinv * imu.dt;
        A.fixed_view_mut::<3, 3>(0, 9).copy_from(&M);

        // Fifth column (wrt accel bias)
        let mut M = -R * imu.dt;
        A.fixed_view_mut::<3, 3>(6, 9).copy_from(&M);
        M *= imu.dt / 2.0;
        A.fixed_view_mut::<3, 3>(3, 9).copy_from(&M);

        A
    }

    #[allow(non_snake_case)]
    pub fn B_Q_BT(&self, imu: &ImuMeasurement, p: &ImuParams) -> Matrix<15, 15> {
        let H = SO3::dexp(self.xi_theta.as_view());
        let Hinv = H.try_inverse().expect("Failed to invert H(theta)");
        let R = SO3::exp(self.xi_theta.as_view()).to_matrix();

        // Construct all partials
        let H_theta_w = Hinv * imu.dt;
        let H_theta_winit = -Hinv * self.dt;

        let H_v_a = R * imu.dt;
        let H_v_ainit = -R * self.dt;

        let H_p_a = H_v_a * imu.dt / 2.0;
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
            gravity: Vector3::<D>::dual_convert(&other.gravity),
        }
    }
}

pub struct ImuMeasurement<D: Numeric = dtype> {
    accel: Vector3<D>,
    gyro: Vector3<D>,
    dt: D,
}

impl<D: Numeric> ops::Sub<ImuBias<D>> for ImuMeasurement<D> {
    type Output = Self;

    fn sub(self, rhs: ImuBias<D>) -> Self {
        ImuMeasurement {
            gyro: self.gyro - rhs.gyro(),
            accel: self.accel - rhs.accel(),
            dt: self.dt,
        }
    }
}

impl<'a, D: Numeric> ops::Sub<&'a ImuBias<D>> for &'a ImuMeasurement<D> {
    type Output = ImuMeasurement<D>;

    fn sub(self, rhs: &'a ImuBias<D>) -> ImuMeasurement<D> {
        ImuMeasurement {
            gyro: self.gyro - rhs.gyro(),
            accel: self.accel - rhs.accel(),
            dt: self.dt,
        }
    }
}

// ----------------------- The actual residual object ----------------------- //
tag_residual!(ImuPreintegrationResidual);

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct ImuPreintegrationResidual {
    delta: ImuDelta,
}

impl Residual6 for ImuPreintegrationResidual {
    type Differ = ForwardProp<Const<15>>;
    type DimIn = Const<30>;
    type DimOut = Const<15>;
    type V1 = SE3;
    type V2 = VectorVar3;
    type V3 = ImuBias;
    type V4 = SE3;
    type V5 = VectorVar3;
    type V6 = ImuBias;

    fn residual6<D: crate::linalg::Numeric>(
        &self,
        x1: SE3<D>,
        v1: VectorVar3<D>,
        b1: ImuBias<D>,
        x2: SE3<D>,
        v2: VectorVar3<D>,
        b2: ImuBias<D>,
    ) -> VectorX<D> {
        // Add dual types to all of our fields
        let delta = ImuDelta::<D>::dual_convert(&self.delta);

        // Pull out the measurements
        let start = ImuState {
            r: x1.rot().clone(),
            v: v1.0,
            p: x1.xyz().into_owned(),
            bias: b1,
        };
        let ImuState {
            r: r2_meas,
            v: v2_meas,
            p: p2_meas,
            bias: b2_meas,
        } = delta.predict(&start);
        let p2_meas = VectorVar3::from(p2_meas);
        let v2_meas = VectorVar3::from(v2_meas);

        // Compute residuals
        let p_2: VectorVar3<D> = x2.xyz().into_owned().into();
        let r_r = r2_meas.ominus(x2.rot());
        let r_vel = v2_meas.ominus(&v2);
        let r_p = p2_meas.ominus(&p_2);
        let r_bias = b2_meas.ominus(&b2);

        let mut residual = VectorX::zeros(15);
        residual.fixed_rows_mut::<3>(0).copy_from(&r_r);
        residual.fixed_rows_mut::<3>(3).copy_from(&r_vel);
        residual.fixed_rows_mut::<3>(6).copy_from(&r_p);
        residual.fixed_rows_mut::<6>(9).copy_from(&r_bias);

        residual
    }
}

impl_residual!(6, ImuPreintegrationResidual);

impl fmt::Display for ImuPreintegrationResidual {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "ImuPreintegrationResidual(theta: {}, v: {}, t: {})",
            self.delta.xi_theta, self.delta.xi_vel, self.delta.xi_pos
        )
    }
}

// ------------------------- The Preintegrator ------------------------- //
#[derive(Clone, Debug)]
pub struct ImuParams {
    // TODO: Remove gravity from here?
    gravity: Vector3,
    cov_accel: Matrix3,
    cov_gyro: Matrix3,
    cov_accel_bias: Matrix3,
    cov_gyro_bias: Matrix3,
    cov_integration: Matrix3,
    cov_winit: Matrix3,
    cov_ainit: Matrix3,
}

/// Implements reasonable parameters for ImuParams including positive gravity
impl Default for ImuParams {
    fn default() -> Self {
        Self {
            gravity: Vector3::new(0.0, 0.0, 9.81),
            cov_accel: Matrix3::identity() * 1e-5,
            cov_gyro: Matrix3::identity() * 1e-5,
            cov_accel_bias: Matrix3::identity() * 1e-6,
            cov_gyro_bias: Matrix3::identity() * 1e-6,
            cov_integration: Matrix3::identity() * 1e-7,
            cov_winit: Matrix3::identity() * 1e-7,
            cov_ainit: Matrix3::identity() * 1e-7,
        }
    }
}

impl ImuParams {
    /// Create a new set of parameters with positive gravity
    pub fn positive() -> Self {
        Self::default()
    }

    /// Create a new set of parameters with negative gravity
    pub fn negative() -> Self {
        let mut params = Self::default();
        params.gravity = -params.gravity;
        params
    }

    pub fn set_scalar_accel(&mut self, val: dtype) {
        self.cov_accel = Matrix3::identity() * val;
    }

    pub fn set_scalar_gyro(&mut self, val: dtype) {
        self.cov_gyro = Matrix3::identity() * val;
    }

    pub fn set_scalar_accel_bias(&mut self, val: dtype) {
        self.cov_accel_bias = Matrix3::identity() * val;
    }

    pub fn set_scalar_gyro_bias(&mut self, val: dtype) {
        self.cov_gyro_bias = Matrix3::identity() * val;
    }

    pub fn set_scalar_integration(&mut self, val: dtype) {
        self.cov_integration = Matrix3::identity() * val;
    }

    // TODO: Do I really need seperate values for this?
    // In practice, I think every makes cov_winit = cov_ainit
    // For now, the public interface assumes they are the same, but behind the
    // scenes we're using both
    pub fn set_scalar_init(&mut self, val: dtype) {
        self.cov_winit = Matrix3::identity() * val;
        self.cov_ainit = Matrix3::identity() * val;
    }
}

/// Performs Imu preintegration
#[derive(Clone, Debug)]
pub struct ImuPreintegrator {
    // Mutable state that will change as we integrate
    delta: ImuDelta,
    cov: Matrix<15, 15>,
    // Constants
    params: ImuParams,
}

impl ImuPreintegrator {
    pub fn new(params: ImuParams, bias_init: ImuBias) -> Self {
        let delta = ImuDelta::new(params.gravity, bias_init);
        Self {
            delta,
            // init with small value to avoid singular matrix
            cov: Matrix::zeros() * 1e-14,
            params,
        }
    }

    #[allow(non_snake_case)]
    pub fn integrate(&mut self, imu: ImuMeasurement) {
        // Construct all matrices before integrating
        let A = self.delta.A(&imu);
        let B_Q_BT = self.delta.B_Q_BT(&imu, &self.params);

        // Update preintegration
        self.delta.integrate(&imu);

        // Update covariance
        self.cov = A * self.cov * A.transpose() + B_Q_BT;

        // Update H
        let Amini = A.fixed_view::<9, 9>(0, 0);
        // This should come from B, turns out it's identical as A
        let Bgyro = A.fixed_view::<9, 3>(0, 9);
        let Baccel = A.fixed_view::<9, 3>(0, 12);
        self.delta.h_bias_gyro = Amini * self.delta.h_bias_gyro + Bgyro;
        self.delta.h_bias_accel = Amini * self.delta.h_bias_accel + Baccel;
    }

    /// Build a corresponding factor
    ///
    /// This consumes the preintegrator and returns a
    /// [factor](crate::factor::Factor) with the proper noise model.
    pub fn build<X1, V1, B1, X2, V2, B2>(
        self,
        x1: X1,
        v1: V1,
        b1: B1,
        x2: X2,
        v2: V2,
        b2: B2,
    ) -> Factor
    where
        X1: TypedSymbol<SE3>,
        V1: TypedSymbol<VectorVar3>,
        B1: TypedSymbol<ImuBias>,
        X2: TypedSymbol<SE3>,
        V2: TypedSymbol<VectorVar3>,
        B2: TypedSymbol<ImuBias>,
    {
        // Create noise from our covariance matrix
        let noise = GaussianNoise::from_matrix_cov(self.cov.as_view());
        let res = ImuPreintegrationResidual { delta: self.delta };
        FactorBuilder::new6(res, x1, v1, b1, x2, v2, b2)
            .noise(noise)
            .build()
    }
}
