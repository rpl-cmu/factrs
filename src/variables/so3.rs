use crate::dtype;
use crate::traits::{DualNum, LieGroup, Variable};
use crate::variables::{Vector3, Vector4, VectorD};
use nalgebra as na;
use std::fmt;
use std::ops;

#[derive(Clone)]
pub struct SO3<D: DualNum<dtype> = dtype> {
    xyzw: Vector4<D>,
}

impl<D: DualNum<dtype>> SO3<D> {
    pub fn from_vec(xyzw: Vector4<D>) -> Self {
        SO3 { xyzw }
    }

    pub fn from_xyzw(x: D, y: D, z: D, w: D) -> Self {
        SO3 {
            xyzw: Vector4::<D>::new(x, y, z, w),
        }
    }

    pub fn from_matrix(mat: &nalgebra::Matrix3<D>) -> Self {
        let trace = mat[(0, 0)].clone() + mat[(1, 1)].clone() + mat[(2, 2)].clone();
        let mut xyzw = Vector4::zeros();
        let zero = D::from(0.0);
        let quarter = D::from(0.25);
        let one = D::from(1.0);
        let two = D::from(2.0);

        if trace > zero {
            let s = D::from(0.5) / (trace + 1.0).sqrt();
            xyzw[3] = quarter / s.clone();
            xyzw[0] = (mat[(2, 1)].clone() - mat[(1, 2)].clone()) * s.clone();
            xyzw[1] = (mat[(0, 2)].clone() - mat[(2, 0)].clone()) * s.clone();
            xyzw[2] = (mat[(1, 0)].clone() - mat[(0, 1)].clone()) * s.clone();
        } else if mat[(0, 0)] > mat[(1, 1)] && mat[(0, 0)] > mat[(2, 2)] {
            let s = two * (one + &mat[(0, 0)] - &mat[(1, 1)] - &mat[(2, 2)]).sqrt();
            xyzw[3] = (mat[(2, 1)].clone() - mat[(1, 2)].clone()) / s.clone();
            xyzw[0] = s.clone() * quarter;
            xyzw[1] = (mat[(0, 1)].clone() + mat[(1, 0)].clone()) / s.clone();
            xyzw[2] = (mat[(0, 2)].clone() + mat[(2, 0)].clone()) / s.clone();
        } else if mat[(1, 1)] > mat[(2, 2)] {
            let s = two * (one + &mat[(1, 1)] - &mat[(0, 0)] - &mat[(2, 2)]).sqrt();
            xyzw[3] = (mat[(0, 2)].clone() - mat[(2, 0)].clone()) / s.clone();
            xyzw[0] = (mat[(0, 1)].clone() + mat[(1, 0)].clone()) / s.clone();
            xyzw[1] = s.clone() * quarter;
            xyzw[2] = (mat[(1, 2)].clone() + mat[(2, 1)].clone()) / s.clone();
        } else {
            let s = two * (one + &mat[(2, 2)] - &mat[(0, 0)] - &mat[(1, 1)]).sqrt();
            xyzw[3] = (mat[(1, 0)].clone() - mat[(0, 1)].clone()) / s.clone();
            xyzw[0] = (mat[(0, 2)].clone() + mat[(2, 0)].clone()) / s.clone();
            xyzw[1] = (mat[(1, 2)].clone() + mat[(2, 1)].clone()) / s.clone();
            xyzw[2] = s.clone() * quarter;
        }

        SO3 { xyzw }
    }

    pub fn to_matrix(&self) -> nalgebra::Matrix3<D> {
        let q0 = self.xyzw[3].clone();
        let q1 = self.xyzw[0].clone();
        let q2 = self.xyzw[1].clone();
        let q3 = self.xyzw[2].clone();

        let mut mat = nalgebra::Matrix3::zeros();
        mat[(0, 0)] = D::from(1.0) - (q2.clone() * q2.clone() + q3.clone() * q3.clone()) * 2.0;
        mat[(0, 1)] = (q1.clone() * q2.clone() - q0.clone() * q3.clone()) * 2.0;
        mat[(0, 2)] = (q1.clone() * q3.clone() + q0.clone() * q2.clone()) * 2.0;
        mat[(1, 0)] = (q1.clone() * q2.clone() + q0.clone() * q3.clone()) * 2.0;
        mat[(1, 1)] = D::from(1.0) - (q1.clone() * q1.clone() + q3.clone() * q3.clone()) * 2.0;
        mat[(1, 2)] = (q2.clone() * q3.clone() - q0.clone() * q1.clone()) * 2.0;
        mat[(2, 0)] = (q1.clone() * q3.clone() - q0.clone() * q2.clone()) * 2.0;
        mat[(2, 1)] = (q2.clone() * q3.clone() + q0.clone() * q1.clone()) * 2.0;
        mat[(2, 2)] = D::from(1.0) - (q1.clone() * q1.clone() + q2.clone() * q2.clone()) * 2.0;

        mat
    }

    pub fn apply(&self, v: &Vector3<D>) -> Vector3<D> {
        // let qv = Self::from_xyzw(v[0], v[1], v[2], (0.0).into());
        // let inv = self.inverse();

        // let v_rot = (&(&inv * &qv) * self).xyzw;
        // Vector3::new(v_rot[0], v_rot[1], v_rot[2])
        v.clone()
    }
}

impl<D: DualNum<dtype>> Variable<D> for SO3<D> {
    const DIM: usize = 3;

    fn identity() -> Self {
        SO3 {
            xyzw: Vector4::new(D::from(0.0), D::from(0.0), D::from(0.0), D::from(1.0)),
        }
    }

    fn inverse(&self) -> Self {
        SO3 {
            xyzw: Vector4::new(
                -self.xyzw[0].clone(),
                -self.xyzw[1].clone(),
                -self.xyzw[2].clone(),
                self.xyzw[3].clone(),
            ),
        }
    }

    fn oplus(&self, delta: &VectorD<D>) -> Self {
        let e = Self::exp(delta);
        self * &e
    }

    fn ominus(&self, other: &Self) -> VectorD<D> {
        (&Variable::inverse(self) * other).log()
    }
}

impl<D: DualNum<dtype>> LieGroup<D> for SO3<D> {
    fn exp(xi: &VectorD<D>) -> Self {
        let mut xyzw = Vector4::zeros();
        let one = D::from(1.0);
        let theta = xi.norm();

        if theta < D::from(1e-2) {
            let theta2 = theta.clone() * theta;
            let six = D::from(6.0);
            xyzw[0] = xi[0].clone() * (D::from(1.0) - theta2.clone() / six.clone());
            xyzw[1] = xi[1].clone() * (D::from(1.0) - theta2.clone() / six.clone());
            xyzw[2] = xi[2].clone() * (D::from(1.0) - theta2.clone() / six.clone());
            xyzw[3] = one;
        } else {
            let theta_half = theta.clone() / D::from(2.0);
            let sin_theta = theta_half.clone().sin();
            xyzw[0] = xi[0].clone() * sin_theta.clone() / theta.clone();
            xyzw[1] = xi[1].clone() * sin_theta.clone() / theta.clone();
            xyzw[2] = xi[2].clone() * sin_theta.clone() / theta.clone();
            xyzw[3] = theta_half.clone().cos();
        }

        SO3 { xyzw }
    }

    fn log(&self) -> VectorD<D> {
        // TODO: Any small angle check needed in here?
        let xi = na::dvector![
            self.xyzw[0].clone(),
            self.xyzw[1].clone(),
            self.xyzw[2].clone()
        ];
        let w = self.xyzw[3].clone();

        let norm_v = xi.norm();
        (xi / norm_v.clone()) * norm_v.atan2(w) * D::from(2.0)
    }

    fn wedge(xi: &VectorD<D>) -> na::DMatrix<D> {
        let mut xi_hat = na::DMatrix::zeros(3, 3);
        xi_hat[(0, 1)] = -xi[2].clone();
        xi_hat[(0, 2)] = xi[1].clone();
        xi_hat[(1, 0)] = xi[2].clone();
        xi_hat[(1, 2)] = -xi[0].clone();
        xi_hat[(2, 0)] = -xi[1].clone();
        xi_hat[(2, 1)] = xi[0].clone();

        xi_hat
    }
}

impl<D: DualNum<dtype>> ops::Mul for SO3<D> {
    type Output = SO3<D>;

    fn mul(self, other: Self) -> Self::Output {
        &self * &other
    }
}

impl<D: DualNum<dtype>> ops::Mul for &SO3<D> {
    type Output = SO3<D>;

    #[rustfmt::skip]
    fn mul(self, other: Self) -> Self::Output {
        let x0 = self.xyzw[0].clone();
        let y0 = self.xyzw[1].clone();
        let z0 = self.xyzw[2].clone();
        let w0 = self.xyzw[3].clone();

        let x1 = other.xyzw[0].clone();
        let y1 = other.xyzw[1].clone();
        let z1 = other.xyzw[2].clone();
        let w1 = other.xyzw[3].clone();

        // Compute the product of the two quaternions, term by term
        let mut xyzw = Vector4::zeros();
        xyzw[0] = w0.clone() * x1.clone() + x0.clone() * w1.clone() + y0.clone() * z1.clone() - z0.clone() * y1.clone();
        xyzw[1] = w0.clone() * y1.clone() - x0.clone() * z1.clone() + y0.clone() * w1.clone() + z0.clone() * x1.clone();
        xyzw[2] = w0.clone() * z1.clone() + x0.clone() * y1.clone() - y0.clone() * x1.clone() + z0.clone() * w1.clone();
        xyzw[3] = w0.clone() * w1.clone() - x0.clone() * x1.clone() - y0.clone() * y1.clone() - z0.clone() * z1.clone();

        SO3 { xyzw }
    }
}

impl<D: DualNum<dtype>> fmt::Display for SO3<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "SO3({:.3}, {:.3}, {:.3}, {:.3})",
            self.xyzw[0], self.xyzw[1], self.xyzw[2], self.xyzw[3]
        )
    }
}

impl<D: DualNum<dtype>> fmt::Debug for SO3<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::DualNum;
    use num_dual::jacobian;

    #[test]
    fn exp_lop() {
        // exp -> log should give back original vector
        let xi = na::dvector![0.1, 0.2, 0.3];
        let so3 = SO3::exp(&xi);
        let log = so3.log();
        println!("xi {xi:?}, {log:?}");
        assert!((xi - log).norm() < 1e-6);
    }

    #[test]
    fn matrix() {
        // to_matrix -> from_matrix should give back original vector
        let xi = na::dvector![0.1, 0.2, 0.3];
        let so3_og = SO3::exp(&xi);
        let mat = so3_og.to_matrix();

        let so3_after = SO3::from_matrix(&mat);
        println!("{:}", so3_og);
        println!("{:}", so3_after);
        assert!((so3_og.xyzw - so3_after.xyzw).norm() < 1e-6);
    }

    #[test]
    fn multiply() {
        // multiply two small x-only angles should give back double angle
        let xi = na::dvector![0.5, 0.0, 0.0];
        let so3 = SO3::exp(&xi);
        let double = &so3 * &so3;
        let xi_double = double.log();
        println!("{:?}", xi_double);
        assert!((xi_double[0] - 1.0) < 1e-6);
    }

    #[test]
    fn inverse() {
        // multiply with inverse should give back identity
        let xi = na::dvector![0.1, 0.2, 0.3];
        let so3 = SO3::<f64>::exp(&xi);
        let so3_inv = so3.inverse();
        let so3_res = &so3 * &so3_inv;
        let id = SO3::<f64>::identity();
        println!("{}", so3_res);
        assert!((so3_res.xyzw - id.xyzw).norm() < 1e-6);
    }

    #[test]
    fn test_jacobian() {
        // Test jacobian of exp(log(x)) = x
        fn compute<D: DualNum<dtype>>(v: VectorD<D>) -> VectorD<D> {
            let so3 = SO3::<D>::exp(&v);
            let mat = so3.to_matrix();
            let so3 = SO3::<D>::from_matrix(&mat);
            so3.log()
        }

        let v = na::dvector![0.1, 0.2, 0.3];
        let (x, dx) = jacobian(compute, v.clone());

        assert!((x - v).norm() < 1e-6);
        assert!((na::DMatrix::identity(3, 3) - dx).norm() < 1e-6);
    }
}
