use std::fmt;

use nalgebra::Const;

use super::{delta::ImuDelta, Accel, Gravity, Gyro, ImuState};
use crate::{
    containers::{Factor, FactorBuilder, TypedSymbol},
    dtype,
    impl_residual,
    linalg::{ForwardProp, Matrix, Matrix3, VectorX},
    noise::GaussianNoise,
    residuals::Residual6,
    tag_residual,
    traits::*,
    variables::{ImuBias, VectorVar3, SE3},
};
// ------------------------- Covariances ------------------------- //

#[derive(Clone, Debug)]
pub struct ImuCovariance {
    pub cov_accel: Matrix3,
    pub cov_gyro: Matrix3,
    pub cov_accel_bias: Matrix3,
    pub cov_gyro_bias: Matrix3,
    pub cov_integration: Matrix3,
    pub cov_winit: Matrix3,
    pub cov_ainit: Matrix3,
}

/// Implements reasonable parameters for ImuCovariance including positive
/// gravity
impl Default for ImuCovariance {
    fn default() -> Self {
        Self {
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

impl ImuCovariance {
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

    // In practice, I think everyone makes cov_winit = cov_ainit
    // For now, the public interface assumes they are the same, but behind the
    // scenes we're using both
    pub fn set_scalar_init(&mut self, val: dtype) {
        self.cov_winit = Matrix3::identity() * val;
        self.cov_ainit = Matrix3::identity() * val;
    }
}

// ------------------------- The Preintegrator ------------------------- //
/// Performs Imu preintegration
#[derive(Clone, Debug)]
pub struct ImuPreintegrator {
    // Mutable state that will change as we integrate
    delta: ImuDelta,
    cov: Matrix<15, 15>,
    // Constants
    params: ImuCovariance,
}

impl ImuPreintegrator {
    pub fn new(params: ImuCovariance, bias_init: ImuBias, gravity: Gravity) -> Self {
        let delta = ImuDelta::new(gravity, bias_init);
        Self {
            delta,
            // init with small value to avoid singular matrix
            cov: Matrix::identity() * 1e-14,
            params,
        }
    }

    #[allow(non_snake_case)]
    pub fn integrate(&mut self, gyro: Gyro, accel: Accel, dt: dtype) {
        // Remove bias estimate
        let gyro = gyro.remove_bias(self.delta.bias_init());
        let accel = accel.remove_bias(self.delta.bias_init());

        // Construct all matrices before integrating
        let A = self.delta.A(&gyro, &accel, dt);
        let B_Q_BT = self.delta.B_Q_BT(&self.params, dt);

        // Update preintegration
        self.delta.integrate(&gyro, &accel, dt);

        // Update covariance
        self.cov = A * self.cov * A.transpose() + B_Q_BT;

        // Update H for bias updates
        self.delta.propagate_H(&A);
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
        // TODO: Does covariance need to be modified as preint will be modified?
        let noise = GaussianNoise::from_matrix_cov(self.cov.as_view());
        let res = ImuPreintegrationResidual { delta: self.delta };
        FactorBuilder::new6(res, x1, v1, b1, x2, v2, b2)
            .noise(noise)
            .build()
    }
}

// ------------------------- The Residual ------------------------- //

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
        let p_2: VectorVar3<D> = x2.xyz().into_owned().into();
        let p2_meas: VectorVar3<D> = p2_meas.into();
        let v2_meas: VectorVar3<D> = v2_meas.into();

        // Compute residuals
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
        write!(f, "ImuPreintegrationResidual({})", self.delta)
    }
}
