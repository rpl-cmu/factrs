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
        write!(f, "SO2({:.3})", self.log()[0])
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

    use crate::{test_lie, test_variable};

    test_variable!(SO2);

    test_lie!(SO2);
}
