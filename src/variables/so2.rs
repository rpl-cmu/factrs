use nalgebra::dmatrix;

use crate::dtype;
use crate::linalg::{dvector, DualNum, DualVec, Matrix2, MatrixX, VectorViewX, VectorX};
use crate::variables::{LieGroup, Variable};
use std::fmt;
use std::ops;

use super::Vector2;

#[derive(Clone)]
pub struct SO2<D: DualNum = dtype> {
    a: D,
    b: D,
}

impl<D: DualNum> SO2<D> {
    pub fn from_theta(theta: D) -> Self {
        SO2 {
            a: theta.clone().cos(),
            b: theta.clone().sin(),
        }
    }

    pub fn from_matrix(mat: &Matrix2<D>) -> Self {
        SO2 {
            a: mat[(0, 0)].clone(),
            b: mat[(1, 0)].clone(),
        }
    }

    pub fn to_matrix(&self) -> Matrix2<D> {
        Matrix2::new(
            self.a.clone(),
            -self.b.clone(),
            self.b.clone(),
            self.a.clone(),
        )
    }

    pub fn apply(&self, v: &Vector2<D>) -> Vector2<D> {
        Vector2::new(
            v[0].clone() * self.a.clone() - v[1].clone() * self.b.clone(),
            v[0].clone() * self.b.clone() + v[1].clone() * self.a.clone(),
        )
    }
}

impl<D: DualNum> Variable<D> for SO2<D> {
    const DIM: usize = 3;
    type Dual = SO2<DualVec>;

    fn identity() -> Self {
        SO2 {
            a: D::from(1.0),
            b: D::from(0.0),
        }
    }

    fn inverse(&self) -> Self {
        SO2 {
            a: self.a.clone(),
            b: -self.b.clone(),
        }
    }

    fn compose(&self, other: &Self) -> Self {
        SO2 {
            a: self.a.clone() * other.a.clone() - self.b.clone() * other.b.clone(),
            b: self.a.clone() * other.b.clone() + self.b.clone() * other.a.clone(),
        }
    }

    fn exp(xi: VectorViewX<D>) -> Self {
        let theta = xi[0].clone();
        SO2::from_theta(theta)
    }

    fn log(&self) -> VectorX<D> {
        dvector![self.b.clone().atan2(self.a.clone())]
    }

    fn dual_self(&self) -> Self::Dual {
        Self::Dual {
            a: self.a.clone().into(),
            b: self.b.clone().into(),
        }
    }
}

impl<D: DualNum> LieGroup<D> for SO2<D> {
    fn hat(xi: VectorViewX<D>) -> MatrixX<D> {
        dmatrix![D::from(0.0), -xi[0].clone(); xi[0].clone(), D::from(0.0)]
    }

    fn adjoint(&self) -> MatrixX<D> {
        MatrixX::identity(2, 2)
    }
}

impl<D: DualNum> ops::Mul for SO2<D> {
    type Output = SO2<D>;

    fn mul(self, other: Self) -> Self::Output {
        self.compose(&other)
    }
}

impl<D: DualNum> ops::Mul for &SO2<D> {
    type Output = SO2<D>;

    fn mul(self, other: Self) -> Self::Output {
        self.compose(other)
    }
}

impl<D: DualNum> fmt::Display for SO2<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "SO2({:.3})", self.log())
    }
}

impl<D: DualNum> fmt::Debug for SO2<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::{Const, DualNum, Dyn};
    use matrixcompare::{assert_matrix_eq, assert_scalar_eq};
    use num_dual::jacobian;

    #[cfg(feature = "f32")]
    pub use std::f32::consts;
    #[cfg(not(feature = "f32"))]
    pub use std::f64::consts;

    #[test]
    fn exp_log() {
        // exp -> log should give back original vector
        let xi = dvector![0.1];
        let so2 = SO2::exp(xi.as_view());
        let log = so2.log();
        println!("xi {xi:?}, {log:?}");
        assert_matrix_eq!(xi, log, comp = float);
    }

    #[test]
    fn matrix() {
        // to_matrix -> from_matrix should give back original vector
        let xi = dvector![0.1];
        let so2_og = SO2::exp(xi.as_view());
        let mat = so2_og.to_matrix();

        let so2_after = SO2::from_matrix(&mat);
        println!("{:}", so2_og);
        println!("{:}", so2_after);
        assert_scalar_eq!(so2_og.a, so2_after.a, comp = float);
        assert_scalar_eq!(so2_og.b, so2_after.b, comp = float);
    }

    #[test]
    fn multiply() {
        // multiply two small x-only angles should give back double angle
        let xi = dvector![0.5];
        let so2 = SO2::exp(xi.as_view());
        let double = &so2 * &so2;
        let xi_double = double.log();
        println!("{:?}", xi_double);
        assert_scalar_eq!(xi_double[0], 1.0, comp = float);
    }

    #[test]
    fn inverse() {
        // multiply with inverse should give back identity
        let xi = dvector![0.1];
        let so2 = SO2::<dtype>::exp(xi.as_view());
        let so2_inv = so2.inverse();
        let so2_res = &so2 * &so2_inv;
        let id = SO2::<dtype>::identity();
        println!("{}", so2_res);
        assert_scalar_eq!(so2_res.a, id.a, comp = float);
        assert_scalar_eq!(so2_res.b, id.b, comp = float);
    }

    #[test]
    fn rotate() {
        // rotate a vector
        let xi = dvector![consts::FRAC_PI_2];
        let so2 = SO2::exp(xi.as_view());
        let v = Vector2::new(1.0, 0.0);
        let v_rot = so2.apply(&v);
        println!("{:?}", v_rot);
        println!("{}", so2.to_matrix());
        println!("{}", so2);
        assert_matrix_eq!(v_rot, Vector2::y(), comp = float);
    }

    #[test]
    fn test_jacobian() {
        // Test jacobian of exp(log(x)) = x
        fn compute<D: DualNum>(v: VectorX<D>) -> VectorX<D> {
            let so2 = SO2::<D>::exp(v.as_view());
            let mat = so2.to_matrix();
            let so2 = SO2::<D>::from_matrix(&mat);
            so2.log()
        }

        let v = dvector![0.1];
        let (x, dx) = jacobian(compute, v.clone());

        assert_matrix_eq!(x, v, comp = float);
        assert_matrix_eq!(MatrixX::identity(1, 1), dx, comp = float);
    }

    // fn var_jacobian<G>(g: G, r: SO2) -> (VectorX<dtype>, MatrixX<dtype>)
    // where
    //     G: FnOnce(SO2<DualVec>) -> VectorX<DualVec>,
    // {
    //     let rot = r.dual(0, r.dim());

    //     let out = g(rot);
    //     let eps = MatrixX::from_rows(
    //         out.map(|res| res.eps.unwrap_generic(Dyn(3), Const::<1>).transpose())
    //             .as_slice(),
    //     );

    //     (out.map(|r| r.re), eps)
    // }

    // TODO: Derive these by hand to check
    // #[test]
    // fn test_jacobian_again() {
    //     fn rotate<D: DualNum>(r: SO2<D>) -> VectorX<D> {
    //         let v = Vector2::new(D::from(1.0), D::from(2.0));
    //         let rotated = r.apply(&v);
    //         dvector![rotated[0].clone(), rotated[1].clone(), rotated[2].clone()]
    //     }

    //     let r = SO2::exp(dvector![0.1, 0.2].as_view());
    //     let (_x, dx) = var_jacobian(rotate, r.clone());

    //     let v = dvector!(1.0, 2.0, 3.0);

    //     #[cfg(not(feature = "left"))]
    //     let dx_exp = -r.to_matrix() * SO2::hat(v.as_view());
    //     #[cfg(feature = "left")]
    //     let dx_exp = -SO2::hat(v.as_view());

    //     println!("Expected: {}", dx_exp);
    //     println!("Actual: {}", dx);

    //     assert_matrix_eq!(dx, dx_exp, comp = float);
    // }
}
