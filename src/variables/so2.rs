use crate::dtype;
use crate::linalg::{
    dvector, Const, DualNum, DualVec, Matrix1, Matrix2, MatrixView, VectorView1, VectorView2,
    VectorViewX, VectorX,
};
use crate::variables::{MatrixLieGroup, Variable};
use std::fmt;
use std::ops;

use super::{Vector1, Vector2};

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
}

impl<D: DualNum> Variable<D> for SO2<D> {
    const DIM: usize = 1;
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

impl<D: DualNum> MatrixLieGroup<D> for SO2<D> {
    type TangentDim = Const<1>;
    type MatrixDim = Const<2>;
    type VectorDim = Const<2>;

    fn adjoint(&self) -> Matrix1<D> {
        Matrix1::identity()
    }

    fn hat(xi: VectorView1<D>) -> Matrix2<D> {
        Matrix2::new(D::from(0.0), -xi[0].clone(), xi[0].clone(), D::from(0.0))
    }

    fn vee(xi: MatrixView<2, 2, D>) -> Vector1<D> {
        Vector1::new(xi[(1, 0)].clone())
    }

    fn hat_swap(xi: VectorView2<D>) -> Vector2<D> {
        Vector2::new(-xi[1].clone(), xi[0].clone())
    }

    fn apply(&self, v: VectorView2<D>) -> Vector2<D> {
        Vector2::new(
            v[0].clone() * self.a.clone() - v[1].clone() * self.b.clone(),
            v[0].clone() * self.b.clone() + v[1].clone() * self.a.clone(),
        )
    }

    fn from_matrix(mat: MatrixView<2, 2, D>) -> Self {
        SO2 {
            a: mat[(0, 0)].clone(),
            b: mat[(1, 0)].clone(),
        }
    }

    fn to_matrix(&self) -> Matrix2<D> {
        Matrix2::new(
            self.a.clone(),
            -self.b.clone(),
            self.b.clone(),
            self.a.clone(),
        )
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
    use crate::linalg::{Diff, DiffResult, DualNum, ForwardProp};
    use matrixcompare::{assert_matrix_eq, assert_scalar_eq};

    #[cfg(feature = "f32")]
    pub use std::f32::consts;
    #[cfg(not(feature = "f32"))]
    pub use std::f64::consts;

    #[test]
    fn matrix() {
        // to_matrix -> from_matrix should give back original vector
        let xi = dvector![0.1];
        let so2_og = SO2::exp(xi.as_view());
        let mat = so2_og.to_matrix();

        let so2_after = SO2::from_matrix(mat.as_view());
        println!("{:}", so2_og);
        println!("{:}", so2_after);
        assert_scalar_eq!(so2_og.a, so2_after.a, comp = float);
        assert_scalar_eq!(so2_og.b, so2_after.b, comp = float);
    }

    #[test]
    fn rotate() {
        // rotate a vector
        let xi = dvector![consts::FRAC_PI_2];
        let so2 = SO2::exp(xi.as_view());
        let v = Vector2::new(1.0, 0.0);
        let v_rot = so2.apply(v.as_view());
        println!("{:?}", v_rot);
        println!("{}", so2.to_matrix());
        println!("{}", so2);
        assert_matrix_eq!(v_rot, Vector2::y(), comp = float);
    }

    // TODO: Analytically derive this one to check
    #[test]
    fn jacobian() {
        fn rotate<D: DualNum>(r: SO2<D>) -> VectorX<D> {
            let v = Vector2::new(D::from(1.0), D::from(2.0));
            let rotated = r.apply(v.as_view());
            dvector![rotated[0].clone(), rotated[1].clone()]
        }

        let r = SO2::exp(dvector![0.1, 0.2].as_view());
        let DiffResult {
            value: _x,
            diff: dx,
        } = ForwardProp::jacobian_1(rotate, &r);

        let v_star = Vector2::new(2.0, -1.0);
        let dx_exp = -r.apply(v_star.as_view());

        println!("Expected: {}", dx_exp);
        println!("Actual: {}", dx);

        assert_matrix_eq!(dx, dx_exp, comp = float);
    }
}
