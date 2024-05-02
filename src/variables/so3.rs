// TODO: Move this to base file, or maybe a core module with all the other traits?
use crate::variables::{LieGroup, Variable};
use nalgebra::{SVector, Vector4};
use std::ops::Mul;

#[derive(Clone)]
pub struct SO3 {
    xyzw: Vector4<f64>,
}

impl Variable for SO3 {
    type TangentVec = SVector<f64, 3>;
    const DIM: usize = 3;

    fn identity() -> Self {
        SO3 {
            xyzw: Vector4::new(0.0, 0.0, 0.0, 1.0),
        }
    }

    fn inverse(&self) -> Self {
        SO3 {
            xyzw: Vector4::new(-self.xyzw[0], -self.xyzw[1], -self.xyzw[2], self.xyzw[3]),
        }
    }

    fn oplus(&self, delta: &Self::TangentVec) -> Self {
        let e = Self::exp(&delta);
        self * &e
    }

    fn ominus(&self, other: &Self) -> Self::TangentVec {
        (&self.inverse() * other).log()
    }
}

impl LieGroup for SO3 {
    fn exp(xi: &Self::TangentVec) -> Self {
        let mut xyzw = Vector4::zeros();

        let theta = xi.norm();
        if theta < 1e-2 {
            xyzw[0] = xi[0] * (1.0 - theta * theta / 6.0);
            xyzw[1] = xi[1] * (1.0 - theta * theta / 6.0);
            xyzw[2] = xi[2] * (1.0 - theta * theta / 6.0);
            xyzw[3] = 1.0;
        } else {
            let theta_half = theta / 2.0;
            let sin_theta = theta_half.sin();
            xyzw[0] = xi[0] * sin_theta / theta;
            xyzw[1] = xi[1] * sin_theta / theta;
            xyzw[2] = xi[2] * sin_theta / theta;
            xyzw[3] = theta_half.cos();
        }

        SO3 { xyzw }
    }

    fn log(&self) -> Self::TangentVec {
        let xi = Self::TangentVec::new(self.xyzw[0], self.xyzw[1], self.xyzw[2]);
        let w = self.xyzw[3];

        let norm_v = xi.norm();
        2.0 * xi * norm_v.atan2(w) / norm_v
    }
}

impl Mul for SO3 {
    type Output = SO3;

    fn mul(self, other: Self) -> SO3 {
        &self * &other
    }
}

impl Mul for &SO3 {
    type Output = SO3;

    fn mul(self, other: Self) -> SO3 {
        let x0 = self.xyzw[0];
        let y0 = self.xyzw[1];
        let z0 = self.xyzw[2];
        let w0 = self.xyzw[3];

        let x1 = other.xyzw[0];
        let y1 = other.xyzw[1];
        let z1 = other.xyzw[2];
        let w1 = other.xyzw[3];

        // Compute the product of the two quaternions, term by term
        let mut xyzw = Vector4::zeros();
        xyzw[0] = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1;
        xyzw[1] = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1;
        xyzw[2] = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1;
        xyzw[3] = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1;

        SO3 { xyzw }
    }
}
