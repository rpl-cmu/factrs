use std::{
    fmt,
    ops::{self, SubAssign},
};

use matrixcompare::Matrix;

use super::VectorVar4;
use crate::{
    dtype,
    linalg::{
        vectorx, AllocatorBuffer, Const, DefaultAllocator, Derivative, DimName, DualAllocator,
        DualVector, Matrix3, MatrixView, Numeric, Vector3, Vector4, VectorDim, VectorView3,
        VectorViewX, VectorX,
    },
    tag_variable,
    variables::{MatrixLieGroup, Variable},
};

tag_variable!(SO3);

/// 3D Special Orthogonal Group
///
/// Implementation of SO(3) for 3D rotations. Specifically, we use quaternions
/// to represent rotations due to their underyling efficiency when computing
/// log/exp maps.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SO3<D: Numeric = dtype> {
    pub xyzw: Vector4<D>,
}

impl<D: Numeric> SO3<D> {
    /// Create a new SO3 from a Vector4
    pub fn from_vec(xyzw: Vector4<D>) -> Self {
        SO3 { xyzw }
    }

    /// Create a new SO3 from x, y, z, w
    pub fn from_xyzw(x: D, y: D, z: D, w: D) -> Self {
        SO3 {
            xyzw: Vector4::<D>::new(x, y, z, w),
        }
    }

    pub fn x(&self) -> D {
        self.xyzw[0]
    }

    pub fn y(&self) -> D {
        self.xyzw[1]
    }

    pub fn z(&self) -> D {
        self.xyzw[2]
    }

    pub fn w(&self) -> D {
        self.xyzw[3]
    }

    pub fn exp(xi: Vector3<D>) -> Self {
        let mut xyzw = Vector4::zeros();

        let theta = xi.norm();

        xyzw.w = (theta * D::from(0.5)).cos();

        if theta < D::from(1e-3) {
            let tmp = xyzw.w * D::from(0.5);
            xyzw.x = xi.x * tmp;
            xyzw.y = xi.y * tmp;
            xyzw.z = xi.z * tmp;
        } else {
            let omega = xi / theta;
            let sin_theta_half = (D::from(1.0) - xyzw.w * xyzw.w).sqrt();
            xyzw.x = omega.x * sin_theta_half;
            xyzw.y = omega.y * sin_theta_half;
            xyzw.z = omega.z * sin_theta_half;
        }

        SO3 { xyzw }
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
            xyzw: Vector4::new(-self.xyzw[0], -self.xyzw[1], -self.xyzw[2], self.xyzw[3]),
        }
    }

    fn compose(&self, other: &Self) -> Self {
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
        xyzw[0] = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1;
        xyzw[1] = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1;
        xyzw[2] = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1;
        xyzw[3] = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1;

        SO3 { xyzw }
    }

    fn exp(xi: VectorViewX<D>) -> Self {
        // TODO figure this out better
        let mut xyzw = Vector4::zeros();
        let theta = xi.norm();

        if theta < D::from(1e-3) {
            xyzw[0] = xi[0] * D::from(0.5);
            xyzw[1] = xi[1] * D::from(0.5);
            xyzw[2] = xi[2] * D::from(0.5);
            xyzw[3] = D::from(1.0);
        } else {
            let theta_half = theta / D::from(2.0);
            let sin_theta = theta_half.sin();
            xyzw[0] = xi[0] * sin_theta / theta;
            xyzw[1] = xi[1] * sin_theta / theta;
            xyzw[2] = xi[2] * sin_theta / theta;
            xyzw[3] = theta_half.cos();
        }

        SO3 { xyzw }
    }

    fn log(&self) -> VectorX<D> {
        let xi = vectorx![self.xyzw[0], self.xyzw[1], self.xyzw[2]];
        // Abs value in case we had a negative quaternion
        let w = self.xyzw[3].abs();

        let norm_v = xi.norm();
        if norm_v < D::from(1e-3) {
            xi * D::from(2.0)
        } else {
            xi * norm_v.atan2(w) * D::from(2.0) / norm_v
        }
    }

    fn dual_convert<DD: Numeric>(other: &Self::Alias<dtype>) -> Self::Alias<DD> {
        SO3 {
            xyzw: VectorVar4::<dtype>::dual_convert(&other.xyzw.into()).into(),
        }
    }

    fn dual_setup<N: DimName>(idx: usize) -> Self::Alias<DualVector<N>>
    where
        AllocatorBuffer<N>: Sync + Send,
        DefaultAllocator: DualAllocator<N>,
        DualVector<N>: Copy,
    {
        let mut x = DualVector::<N>::from_re(0.0);
        let mut eps = VectorDim::<N>::zeros();
        eps[idx] = 0.5;
        x.eps = Derivative::new(Some(eps));

        let mut y = DualVector::<N>::from_re(0.0);
        let mut eps = VectorDim::<N>::zeros();
        eps[idx + 1] = 0.5;
        y.eps = Derivative::new(Some(eps));

        let mut z = DualVector::<N>::from_re(0.0);
        let mut eps = VectorDim::<N>::zeros();
        eps[idx + 2] = 0.5;
        z.eps = Derivative::new(Some(eps));

        let w = DualVector::<N>::from_re(1.0);

        SO3::from_xyzw(x, y, z, w)
    }
}

impl<D: Numeric> MatrixLieGroup<D> for SO3<D> {
    type TangentDim = Const<3>;
    type MatrixDim = Const<3>;
    type VectorDim = Const<3>;

    fn adjoint(&self) -> Matrix3<D> {
        let q0 = self.xyzw[3];
        let q1 = self.xyzw[0];
        let q2 = self.xyzw[1];
        let q3 = self.xyzw[2];

        // Same as to_matrix function of SO3 -> Just avoiding copying from Matrix3 to
        // MatrixD
        let mut mat = Matrix3::zeros();
        mat[(0, 0)] = D::from(1.0) - (q2 * q2 + q3 * q3) * 2.0;
        mat[(0, 1)] = (q1 * q2 - q0 * q3) * 2.0;
        mat[(0, 2)] = (q1 * q3 + q0 * q2) * 2.0;
        mat[(1, 0)] = (q1 * q2 + q0 * q3) * 2.0;
        mat[(1, 1)] = D::from(1.0) - (q1 * q1 + q3 * q3) * 2.0;
        mat[(1, 2)] = (q2 * q3 - q0 * q1) * 2.0;
        mat[(2, 0)] = (q1 * q3 - q0 * q2) * 2.0;
        mat[(2, 1)] = (q2 * q3 + q0 * q1) * 2.0;
        mat[(2, 2)] = D::from(1.0) - (q1 * q1 + q2 * q2) * 2.0;

        mat
    }

    fn hat(xi: VectorView3<D>) -> Matrix3<D> {
        let mut xi_hat = Matrix3::zeros();
        xi_hat[(0, 1)] = -xi[2];
        xi_hat[(0, 2)] = xi[1];
        xi_hat[(1, 0)] = xi[2];
        xi_hat[(1, 2)] = -xi[0];
        xi_hat[(2, 0)] = -xi[1];
        xi_hat[(2, 1)] = xi[0];

        xi_hat
    }

    fn vee(xi: MatrixView<3, 3, D>) -> Vector3<D> {
        Vector3::new(xi[(2, 1)], xi[(0, 2)], xi[(1, 0)])
    }

    fn hat_swap(xi: VectorView3<D>) -> Matrix3<D> {
        -Self::hat(xi)
    }

    fn from_matrix(mat: MatrixView<3, 3, D>) -> Self {
        let trace = mat[(0, 0)] + mat[(1, 1)] + mat[(2, 2)];
        let mut xyzw = Vector4::zeros();
        let zero = D::from(0.0);
        let quarter = D::from(0.25);
        let one = D::from(1.0);
        let two = D::from(2.0);

        if trace > zero {
            let s = D::from(0.5) / (trace + 1.0).sqrt();
            xyzw[3] = quarter / s;
            xyzw[0] = (mat[(2, 1)] - mat[(1, 2)]) * s;
            xyzw[1] = (mat[(0, 2)] - mat[(2, 0)]) * s;
            xyzw[2] = (mat[(1, 0)] - mat[(0, 1)]) * s;
        } else if mat[(0, 0)] > mat[(1, 1)] && mat[(0, 0)] > mat[(2, 2)] {
            let s = two * (one + mat[(0, 0)] - mat[(1, 1)] - mat[(2, 2)]).sqrt();
            xyzw[3] = (mat[(2, 1)] - mat[(1, 2)]) / s;
            xyzw[0] = s * quarter;
            xyzw[1] = (mat[(0, 1)] + mat[(1, 0)]) / s;
            xyzw[2] = (mat[(0, 2)] + mat[(2, 0)]) / s;
        } else if mat[(1, 1)] > mat[(2, 2)] {
            let s = two * (one + mat[(1, 1)] - mat[(0, 0)] - mat[(2, 2)]).sqrt();
            xyzw[3] = (mat[(0, 2)] - mat[(2, 0)]) / s;
            xyzw[0] = (mat[(0, 1)] + mat[(1, 0)]) / s;
            xyzw[1] = s * quarter;
            xyzw[2] = (mat[(1, 2)] + mat[(2, 1)]) / s;
        } else {
            let s = two * (one + mat[(2, 2)] - mat[(0, 0)] - mat[(1, 1)]).sqrt();
            xyzw[3] = (mat[(1, 0)] - mat[(0, 1)]) / s;
            xyzw[0] = (mat[(0, 2)] + mat[(2, 0)]) / s;
            xyzw[1] = (mat[(1, 2)] + mat[(2, 1)]) / s;
            xyzw[2] = s * quarter;
        }

        SO3 { xyzw }
    }

    fn to_matrix(&self) -> Matrix3<D> {
        let q0 = self.xyzw[3];
        let q1 = self.xyzw[0];
        let q2 = self.xyzw[1];
        let q3 = self.xyzw[2];

        let mut mat = Matrix3::zeros();
        mat[(0, 0)] = D::from(1.0) - (q2 * q2 + q3 * q3) * 2.0;
        mat[(0, 1)] = (q1 * q2 - q0 * q3) * 2.0;
        mat[(0, 2)] = (q1 * q3 + q0 * q2) * 2.0;
        mat[(1, 0)] = (q1 * q2 + q0 * q3) * 2.0;
        mat[(1, 1)] = D::from(1.0) - (q1 * q1 + q3 * q3) * 2.0;
        mat[(1, 2)] = (q2 * q3 - q0 * q1) * 2.0;
        mat[(2, 0)] = (q1 * q3 - q0 * q2) * 2.0;
        mat[(2, 1)] = (q2 * q3 + q0 * q1) * 2.0;
        mat[(2, 2)] = D::from(1.0) - (q1 * q1 + q2 * q2) * 2.0;

        mat
    }

    fn apply(&self, v: VectorView3<D>) -> Vector3<D> {
        let qv = Self::from_xyzw(v[0], v[1], v[2], (0.0).into());
        let inv = self.inverse();

        let v_rot = (&(self * &qv) * &inv).xyzw;
        Vector3::new(v_rot[0], v_rot[1], v_rot[2])
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
