use core::fmt;

use nalgebra::Const;

use super::Residual6;
use crate::{
    containers::{Factor, FactorBuilder, TypedSymbol},
    dtype,
    impl_residual,
    linalg::{ForwardProp, Matrix, Matrix3, Vector, Vector3, VectorX},
    noise::GaussianNoise,
    tag_residual,
    traits::*,
    variables::{ImuBias, MatrixLieGroup, VectorVar3, SE3, SO3},
};

// ----------------------- The actual residual object ----------------------- //
tag_residual!(ImuPreintegrationResidual);

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct ImuPreintegrationResidual {
    preint: Vector<9>,
    gravity: Vector3,
    dt: dtype,
    bias_hat: ImuBias,
    h_bias_accel: Matrix<9, 3>,
    h_bias_gyro: Matrix<9, 3>,
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
        let preint = Vector::<9, dtype>::dual_convert::<D>(&self.preint);
        let bias_hat = ImuBias::<dtype>::dual_convert::<D>(&self.bias_hat);
        let g = Vector3::<dtype>::dual_convert::<D>(&self.gravity);
        let dt = D::from(self.dt);
        let h_bias_accel = Matrix::<9, 3, dtype>::dual_convert::<D>(&self.h_bias_accel);
        let h_bias_gyro = Matrix::<9, 3, dtype>::dual_convert::<D>(&self.h_bias_gyro);

        // Compute first-order updates to preint with bias
        let bias_diff = &b1 - &bias_hat;
        let preint = preint + h_bias_accel * bias_diff.accel() + h_bias_gyro * bias_diff.gyro();

        // Split preint into components
        let xi_theta = preint.fixed_rows::<3>(0);
        let xi_v = preint.fixed_rows::<3>(3).into_owned();
        let xi_t = preint.fixed_rows::<3>(6).into_owned();

        let x1_r = x1.rot();
        let x1_t = x1.xyz();

        // Estimate x2
        // R2_meas = R1 * exp(xi_theta)
        let r2_meas = x1_r.compose(&SO3::exp(xi_theta.as_view()));
        // v2_meas = v1 + g * dt + R1 * xi_v
        let v2_meas = v1.0 + g * dt + x1_r.apply(xi_v.as_view());
        let v2_meas: VectorVar3<D> = v2_meas.into();
        // p2_meas = p1 + v1 * dt + 0.5 * g * dt^2 + R1 * delta_t
        let p2_meas = x1_t + v1.0 * dt + g * dt * dt * D::from(0.5) + x1_r.apply(xi_t.as_view());
        let p2_meas: VectorVar3<D> = p2_meas.into();
        let b2_meas = b1;

        // Compute residual
        let p_2: VectorVar3<D> = x2.xyz().into_owned().into();
        let r_r = r2_meas.ominus(x2.rot());
        let r_p = p2_meas.ominus(&p_2);
        let r_vel = v2_meas.ominus(&v2);
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
        write!(f, "ImuPreintegrationResidual(preint: {})", self.preint)
    }
}

// ------------------------- The Preintegrator ------------------------- //
#[derive(Clone, Debug)]
pub struct ImuParams {
    g: Vector3,
    cov_accel: Matrix3,
    cov_gyro: Matrix3,
    cov_accel_bias: Matrix3,
    cov_gyro_bias: Matrix3,
    cov_integration: Matrix3,
    cov_init: Matrix3,
}

/// Implements reasonable parameters for ImuParams including positive gravity
impl Default for ImuParams {
    fn default() -> Self {
        Self {
            g: Vector3::new(0.0, 0.0, 9.81),
            cov_accel: Matrix3::identity() * 1e-5,
            cov_gyro: Matrix3::identity() * 1e-5,
            cov_accel_bias: Matrix3::identity() * 1e-6,
            cov_gyro_bias: Matrix3::identity() * 1e-6,
            cov_integration: Matrix3::identity() * 1e-7,
            cov_init: Matrix3::identity() * 1e-7,
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
        params.g = -params.g;
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

    pub fn set_scalar_init(&mut self, val: dtype) {
        self.cov_init = Matrix3::identity() * val;
    }
}

pub struct ImuPreint {
    delta_t: dtype,
    delta_r: Vector3,
    delta_v: Vector3,
    delta_p: Vector3,
    bias_hat: ImuBias,
    h_bias_accel: Matrix<9, 3>,
    h_bias_gyro: Matrix<9, 3>,
    gravity: Vector3,
}

#[derive(Clone, Debug)]
pub struct ImuPreintegrator {
    // Mutable state that will change as we integrate
    // TODO: Combine these into a struct? I pass them all into the residual anyways
    delta_t: dtype,
    delta_r: Vector3,
    delta_v: Vector3,
    delta_p: Vector3,
    bias_hat: ImuBias,
    h_bias_accel: Matrix<9, 3>,
    h_bias_gyro: Matrix<9, 3>,
    cov: Matrix<15, 15>,
    // Constants
    params: ImuParams,
}

impl ImuPreintegrator {
    pub fn new(params: ImuParams, bias_hat: ImuBias) -> Self {
        Self {
            delta_t: 0.0,
            delta_r: Vector3::zeros(),
            delta_v: Vector3::zeros(),
            delta_p: Vector3::zeros(),
            bias_hat,
            h_bias_accel: Matrix::zeros(),
            h_bias_gyro: Matrix::zeros(),
            // init with small value to avoid singular matrix
            cov: Matrix::zeros() * 1e-12,
            params,
        }
    }

    #[allow(non_snake_case)]
    pub fn integrate(&mut self, dt: dtype, accel: Vector3, gyro: Vector3) {
        // Update preint
        self.delta_t += dt;
        let accel = accel - self.bias_hat.accel();
        let gyro = gyro - self.bias_hat.gyro();
        let r_kaccel = SO3::exp(self.delta_r.as_view()).apply(accel.as_view());
        let H = SO3::dexp(self.delta_r.as_view());
        let Hinv = H.try_inverse().expect("Failed to invert H(theta)");

        self.delta_r += Hinv * gyro * dt;
        self.delta_v += r_kaccel * dt;
        self.delta_p += self.delta_v * dt + 0.5 * r_kaccel * dt * dt;

        // Update H

        // Update covariance
        // TODO: Need to verify dimensions of all these
        let R = SO3::exp(self.delta_r.as_view()).to_matrix();
        let mut F = Matrix::<15, 15>::identity();
        let A = Matrix3::identity() - SO3::hat(gyro.as_view()) * dt / 2.0;
        F.fixed_view_mut::<3, 3>(0, 0).copy_from(&A);
        let A = Hinv * dt;
        F.fixed_view_mut::<3, 3>(0, 12).copy_from(&A);
        let mut A = -R * SO3::hat(accel.as_view()) * Hinv * dt;
        F.fixed_view_mut::<3, 3>(6, 0).copy_from(&A);
        A *= dt / 2.0;
        F.fixed_view_mut::<3, 3>(3, 0).copy_from(&A);
        let A = Matrix3::identity() * dt;
        F.fixed_view_mut::<3, 3>(3, 6).copy_from(&A);
        let mut A = R * dt;
        F.fixed_view_mut::<3, 3>(6, 9).copy_from(&A);
        A *= dt / 2.0;
        F.fixed_view_mut::<3, 3>(3, 9).copy_from(&A);

        // TODO: Fill out this beast
        let G_Q_Gt = Matrix::<15, 15>::identity();
        self.cov = F * self.cov * F.transpose() + G_Q_Gt;
    }

    /// Build a corresponding factor
    ///
    /// This consumes the preintegrator and returns a
    /// [factor](crate::factor::Factor) that can be inserted into a factor
    /// graph.
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

        // Copy preint into a single vector
        let mut preint: Vector<9, dtype> = Vector::zeros();
        preint.fixed_rows_mut::<3>(0).copy_from(&self.delta_r);
        preint.fixed_rows_mut::<3>(3).copy_from(&self.delta_v);
        preint.fixed_rows_mut::<3>(6).copy_from(&self.delta_p);

        let res = ImuPreintegrationResidual {
            preint,
            gravity: self.params.g,
            dt: self.delta_t,
            bias_hat: self.bias_hat,
            h_bias_accel: self.h_bias_accel,
            h_bias_gyro: self.h_bias_gyro,
        };

        FactorBuilder::new6(res, x1, v1, b1, x2, v2, b2)
            .noise(noise)
            .build()
    }
}
