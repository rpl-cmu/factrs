use crate::dtype;
use crate::linalg::{dvector, Matrix3, Matrix4, MatrixX, Vector3, VectorX};
use crate::traits::{DualNum, DualVec, LieGroup, Variable};
use crate::variables::SO3;
use std::fmt;
use std::ops;

#[derive(Clone, Debug)]
pub struct SE3<D: DualNum = dtype> {
    rot: SO3<D>,
    xyz: Vector3<D>,
}

impl<D: DualNum> SE3<D> {
    pub fn to_matrix(&self) -> Matrix4<D> {
        let mut mat = Matrix4::<D>::identity();
        mat.fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&self.rot.to_matrix());
        mat.fixed_view_mut::<3, 1>(0, 3).copy_from(&self.xyz);
        mat
    }

    pub fn from_matrix(mat: &Matrix4<D>) -> Self {
        let rot = mat.fixed_view::<3, 3>(0, 0).into();
        let rot = SO3::from_matrix(&rot);

        let xyz = mat.fixed_view::<3, 1>(0, 3).into();

        SE3 { rot, xyz }
    }
}

impl<D: DualNum> Variable<D> for SE3<D> {
    const DIM: usize = 3;
    type Dual = SE3<DualVec>;

    fn identity() -> Self {
        SE3 {
            rot: SO3::<D>::identity(),
            xyz: Variable::identity(),
        }
    }

    fn inverse(&self) -> Self {
        SE3 {
            rot: self.rot.inverse(),
            xyz: -&self.rot.apply(&self.xyz),
        }
    }

    fn oplus(&self, delta: &VectorX<D>) -> Self {
        let e = Self::exp(delta);
        self * &e
    }

    fn ominus(&self, other: &Self) -> VectorX<D> {
        println!("self: {}", self);
        println!("other: {}", other);
        let delta = &self.inverse() * other;
        println!("delta: {}", delta);
        (&Variable::inverse(self) * other).log()
    }

    fn dual_self(&self) -> Self::Dual {
        SE3 {
            rot: self.rot.dual_self(),
            xyz: self.xyz.dual_self(),
        }
    }
}

impl<D: DualNum> LieGroup<D> for SE3<D> {
    // TODO: Both of this functions need to be tested!
    #[allow(non_snake_case)]
    fn exp(xi: &VectorX<D>) -> Self {
        let xi_rot = dvector![xi[0].clone(), xi[1].clone(), xi[2].clone()];
        let xyz = Vector3::new(xi[3].clone(), xi[4].clone(), xi[5].clone());

        let w = xi_rot.norm();
        let rot = SO3::<D>::exp(&xi_rot);

        let I = Matrix3::identity();
        let wx = SO3::wedge(&xi_rot);
        let V = if w.clone() < D::from(1e-3) {
            I + &wx / D::from(2.0) + &wx * &wx / D::from(6.0) + &wx * &wx * &wx / D::from(24.0)
        } else {
            let A = w.clone().sin() / w.clone();
            let B = (D::from(1.0) - w.clone().cos()) / (w.clone() * w.clone());
            let C = (D::from(1.0) - A) / (w.clone() * w.clone());

            I + &wx * &wx * B + &wx * &wx * &wx * C
        };

        SE3 { rot, xyz: V * xyz }
    }

    #[allow(non_snake_case)]
    fn log(&self) -> VectorX<D> {
        let mut xi = VectorX::zeros(6);
        let xi_theta = self.rot.log();
        xi.as_mut_slice()[0..3].clone_from_slice(xi_theta.as_slice());

        let w = xi.norm();
        let I = Matrix3::identity();
        let wx = SO3::wedge(&xi_theta);

        let V: Matrix3<D> = if w.clone().abs() < D::from(1e-6) {
            I + &wx / D::from(2.0) + &wx * &wx / D::from(6.0) + &wx * &wx * &wx / D::from(24.0)
        } else {
            let A = w.clone().sin() / w.clone();
            let B = (D::from(1.0) - w.clone().cos()) / (w.clone() * w.clone());
            let C = (D::from(1.0) - A) / (w.clone() * w.clone());

            I + &wx * &wx * B + &wx * &wx * &wx * C
        };

        let Vinv = V.try_inverse().unwrap();
        let xyz = Vinv * self.xyz.clone();

        xi.as_mut_slice()[3..6].clone_from_slice(xyz.as_slice());

        xi
    }

    fn wedge(xi: &VectorX<D>) -> MatrixX<D> {
        let mut mat = MatrixX::<D>::zeros(4, 4);
        mat[(0, 1)] = -xi[2].clone();
        mat[(0, 2)] = xi[1].clone();
        mat[(1, 0)] = xi[2].clone();
        mat[(1, 2)] = -xi[0].clone();
        mat[(2, 0)] = -xi[1].clone();
        mat[(2, 1)] = xi[0].clone();

        mat[(0, 3)] = xi[3].clone();
        mat[(1, 3)] = xi[4].clone();
        mat[(2, 3)] = xi[5].clone();

        mat
    }
}

impl<D: DualNum> ops::Mul for SE3<D> {
    type Output = SE3<D>;

    fn mul(self, other: Self) -> Self::Output {
        &self * &other
    }
}

impl<D: DualNum> ops::Mul for &SE3<D> {
    type Output = SE3<D>;

    fn mul(self, other: Self) -> Self::Output {
        SE3 {
            rot: &self.rot * &other.rot,
            xyz: self.rot.apply(&other.xyz) + self.xyz.clone(),
        }
    }
}

impl<D: DualNum> fmt::Display for SE3<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} {:?}", self.rot, self.xyz)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::DualNum;
    use num_dual::jacobian;

    #[test]
    fn exp_log() {
        let xi = dvector![0.1, 0.2, 0.3, 1.0, 2.0, 3.0];
        let se3 = SE3::exp(&xi);
        let xi_hat = se3.log();
        println!("{} {}", xi, xi_hat);
        assert!((xi_hat - xi).norm() < 1e-6);
    }

    #[test]
    fn matrix() {
        // to_matrix -> from_matrix shoudl give back original vector
        let xi = dvector![0.1, 0.2, 0.3, 1.0, 2.0, 3.0];
        let se3 = SE3::exp(&xi);
        let mat = se3.to_matrix();

        let se3_hat = SE3::from_matrix(&mat);

        assert!(se3.ominus(&se3_hat).norm() < 1e-6);
    }

    #[test]
    fn inverse() {
        let xi = dvector![0.1, 0.2, 0.3, 1.0, 2.0, 3.0];
        let se3 = SE3::exp(&xi);
        let se3_inv = se3.inverse();

        let out = &se3 * &se3_inv;
        // println!("{}", out);
        // println!("{}", out.inverse());
        // println!("{:?}", out.ominus(&SE3::identity()));
        assert!(out.ominus(&SE3::identity()).norm() < 1e-6);
    }

    #[test]
    fn test_jacobian() {
        // Test jacobian of exp(log(x)) = x
        fn compute<D: DualNum>(v: VectorX<D>) -> VectorX<D> {
            let se3 = SE3::<D>::exp(&v);
            let mat = se3.to_matrix();
            let se3 = SE3::<D>::from_matrix(&mat);
            se3.log()
        }

        let v = dvector![0.1, 0.2, 0.3, 1.0, 2.0, 3.0];
        let (x, dx) = jacobian(compute, v.clone());

        assert!((x - v).norm() < 1e-6);
        assert!((MatrixX::identity(6, 6) - dx).norm() < 1e-6);
    }
}
