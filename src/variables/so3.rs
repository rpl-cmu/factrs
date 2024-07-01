use crate::{
    dtype,
    linalg::{
        dvector, Const, DualVectorX, Matrix3, MatrixView, Numeric, Vector3, Vector4, VectorView3,
        VectorViewX, VectorX,
    },
    variables::{MatrixLieGroup, Variable},
};
use std::{fmt, ops};

#[derive(Clone)]
pub struct SO3<D: Numeric = dtype> {
    pub xyzw: Vector4<D>,
}

impl<D: Numeric> SO3<D> {
    pub fn from_vec(xyzw: Vector4<D>) -> Self {
        SO3 { xyzw }
    }

    pub fn from_xyzw(x: D, y: D, z: D, w: D) -> Self {
        SO3 {
            xyzw: Vector4::<D>::new(x, y, z, w),
        }
    }
}

impl<D: Numeric> Variable<D> for SO3<D> {
    type Dim = Const<3>;
    type Alias<DD: Numeric> = SO3<DD>;

    fn identity() -> Self {
        SO3 { xyzw: Vector4::w() }
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

    fn compose(&self, other: &Self) -> Self {
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
        xyzw[0] = w0.clone() * x1.clone() + x0.clone() * w1.clone() + y0.clone() * z1.clone()
            - z0.clone() * y1.clone();
        xyzw[1] = w0.clone() * y1.clone() - x0.clone() * z1.clone()
            + y0.clone() * w1.clone()
            + z0.clone() * x1.clone();
        xyzw[2] = w0.clone() * z1.clone() + x0.clone() * y1.clone() - y0.clone() * x1.clone()
            + z0.clone() * w1.clone();
        xyzw[3] = w0.clone() * w1.clone()
            - x0.clone() * x1.clone()
            - y0.clone() * y1.clone()
            - z0.clone() * z1.clone();

        SO3 { xyzw }
    }

    fn exp(xi: VectorViewX<D>) -> Self {
        let mut xyzw = Vector4::zeros();
        let theta = xi.norm();

        if theta < D::from(1e-3) {
            xyzw[0] = xi[0].clone() * D::from(0.5);
            xyzw[1] = xi[1].clone() * D::from(0.5);
            xyzw[2] = xi[2].clone() * D::from(0.5);
            xyzw[3] = D::from(1.0);
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

    fn log(&self) -> VectorX<D> {
        let xi = dvector![
            self.xyzw[0].clone(),
            self.xyzw[1].clone(),
            self.xyzw[2].clone()
        ];
        let w = self.xyzw[3].clone();

        let norm_v = xi.norm();
        if norm_v < D::from(1e-3) {
            xi * D::from(2.0)
        } else {
            xi * norm_v.clone().atan2(w) * D::from(2.0) / norm_v
        }
    }

    fn dual_convert<DD: Numeric>(other: &Self::Alias<dtype>) -> Self::Alias<DD> {
        SO3 {
            xyzw: Vector4::<dtype>::dual_convert(&other.xyzw),
        }
    }
}

impl<D: Numeric> MatrixLieGroup<D> for SO3<D> {
    type TangentDim = Const<3>;
    type MatrixDim = Const<3>;
    type VectorDim = Const<3>;

    fn adjoint(&self) -> Matrix3<D> {
        let q0 = self.xyzw[3].clone();
        let q1 = self.xyzw[0].clone();
        let q2 = self.xyzw[1].clone();
        let q3 = self.xyzw[2].clone();

        // Same as to_matrix function of SO3 -> Just avoiding copying from Matrix3 to MatrixD
        let mut mat = Matrix3::zeros();
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

    fn hat(xi: VectorView3<D>) -> Matrix3<D> {
        let mut xi_hat = Matrix3::zeros();
        xi_hat[(0, 1)] = -xi[2].clone();
        xi_hat[(0, 2)] = xi[1].clone();
        xi_hat[(1, 0)] = xi[2].clone();
        xi_hat[(1, 2)] = -xi[0].clone();
        xi_hat[(2, 0)] = -xi[1].clone();
        xi_hat[(2, 1)] = xi[0].clone();

        xi_hat
    }

    fn vee(xi: MatrixView<3, 3, D>) -> Vector3<D> {
        Vector3::new(xi[(2, 1)].clone(), xi[(0, 2)].clone(), xi[(1, 0)].clone())
    }

    fn hat_swap(xi: VectorView3<D>) -> Matrix3<D> {
        -Self::hat(xi)
    }

    fn from_matrix(mat: MatrixView<3, 3, D>) -> Self {
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

    fn to_matrix(&self) -> Matrix3<D> {
        let q0 = self.xyzw[3].clone();
        let q1 = self.xyzw[0].clone();
        let q2 = self.xyzw[1].clone();
        let q3 = self.xyzw[2].clone();

        let mut mat = Matrix3::zeros();
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

    fn apply(&self, v: VectorView3<D>) -> Vector3<D> {
        let qv = Self::from_xyzw(v[0].clone(), v[1].clone(), v[2].clone(), (0.0).into());
        let inv = self.inverse();

        let v_rot = (&(self * &qv) * &inv).xyzw;
        Vector3::new(v_rot[0].clone(), v_rot[1].clone(), v_rot[2].clone())
    }
}

impl<D: Numeric> ops::Mul for SO3<D> {
    type Output = SO3<D>;

    fn mul(self, other: Self) -> Self::Output {
        self.compose(&other)
    }
}

impl<D: Numeric> ops::Mul for &SO3<D> {
    type Output = SO3<D>;

    fn mul(self, other: Self) -> Self::Output {
        self.compose(other)
    }
}

impl<D: Numeric> fmt::Display for SO3<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "SO3({:.3}, {:.3}, {:.3}, {:.3})",
            self.xyzw[0], self.xyzw[1], self.xyzw[2], self.xyzw[3]
        )
    }
}

impl<D: Numeric> fmt::Debug for SO3<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{test_lie, test_variable};

    test_variable!(SO3);

    test_lie!(SO3);
}
