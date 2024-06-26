use nalgebra::dvector;

use crate::dtype;
use crate::linalg::{
    Const, DualNum, DualVec, Matrix2, Matrix2x3, Matrix3, MatrixView, Vector2, VectorView2,
    VectorView3, VectorViewX, VectorX,
};
use crate::variables::{MatrixLieGroup, Variable, SO2};
use std::fmt;
use std::ops;

use super::Vector3;

#[derive(Clone)]
pub struct SE2<D: DualNum = dtype> {
    rot: SO2<D>,
    xy: Vector2<D>,
}

impl<D: DualNum> SE2<D> {
    pub fn new(theta: D, x: D, y: D) -> Self {
        SE2 {
            rot: SO2::from_theta(theta),
            xy: Vector2::new(x, y),
        }
    }

    pub fn x(&self) -> D {
        self.xy[0].clone()
    }

    pub fn y(&self) -> D {
        self.xy[1].clone()
    }

    pub fn theta(&self) -> D {
        self.rot.log()[0].clone()
    }
}

impl<D: DualNum> Variable<D> for SE2<D> {
    const DIM: usize = 3;
    type Dual = SE2<DualVec>;

    fn identity() -> Self {
        SE2 {
            rot: Variable::identity(),
            xy: Variable::identity(),
        }
    }

    fn compose(&self, other: &Self) -> Self {
        SE2 {
            rot: &self.rot * &other.rot,
            xy: self.rot.apply(other.xy.as_view()) + &self.xy,
        }
    }

    fn inverse(&self) -> Self {
        let inv = self.rot.inverse();
        SE2 {
            xy: -&inv.apply(self.xy.as_view()),
            rot: inv,
        }
    }

    #[allow(non_snake_case)]
    #[allow(clippy::needless_borrow)]
    fn exp(xi: VectorViewX<D>) -> Self {
        let theta = xi[0].clone();
        let xy = Vector2::new(xi[1].clone(), xi[2].clone());

        let rot = SO2::<D>::exp(xi.rows(0, 1));

        let xy = if cfg!(feature = "fake_exp") {
            xy
        } else {
            let A;
            let B;
            if theta < D::from(1e-5) {
                A = D::from(1.0);
                B = D::from(0.0);
            } else {
                A = (&theta).sin() / (&theta);
                B = (D::from(1.0) - (&theta).cos()) / (&theta);
            };
            let V = Matrix2::new(A.clone(), -B.clone(), B, A);
            V * xy
        };

        SE2 { rot, xy }
    }

    #[allow(non_snake_case)]
    #[allow(clippy::needless_borrow)]
    fn log(&self) -> VectorX<D> {
        let theta = self.rot.log()[0].clone();

        let xy = if cfg!(feature = "fake_exp") {
            &self.xy
        } else {
            let A;
            let B;
            if theta < D::from(1e-5) {
                A = D::from(1.0);
                B = D::from(0.0);
            } else {
                A = (&theta).sin() / (&theta);
                B = (D::from(1.0) - (&theta).cos()) / (&theta);
            };
            let V = Matrix2::new(A.clone(), -B.clone(), B, A);

            let Vinv = V.try_inverse().expect("V is not invertible");
            &(&Vinv * &self.xy)
        };

        dvector![theta, xy[0].clone(), xy[1].clone()]
    }

    fn dual_self(&self) -> Self::Dual {
        SE2 {
            rot: self.rot.dual_self(),
            xy: self.xy.dual_self(),
        }
    }
}

impl<D: DualNum> MatrixLieGroup<D> for SE2<D> {
    type TangentDim = Const<3>;
    type MatrixDim = Const<3>;
    type VectorDim = Const<2>;

    fn adjoint(&self) -> Matrix3<D> {
        let mut mat = Matrix3::<D>::zeros();

        let r_mat = self.rot.to_matrix();

        mat.fixed_view_mut::<2, 2>(0, 0).copy_from(&r_mat);
        mat[(0, 2)] = self.xy[2].clone();
        mat[(1, 2)] = -self.xy[1].clone();

        mat
    }

    fn hat(xi: VectorView3<D>) -> Matrix3<D> {
        let mut mat = Matrix3::<D>::zeros();
        mat[(0, 1)] = -xi[0].clone();
        mat[(1, 0)] = xi[0].clone();

        mat[(0, 2)] = xi[1].clone();
        mat[(1, 2)] = xi[2].clone();

        mat
    }

    fn vee(xi: MatrixView<3, 3, D>) -> Vector3<D> {
        Vector3::new(xi[(1, 0)].clone(), xi[(0, 1)].clone(), xi[(0, 2)].clone())
    }

    fn apply(&self, v: VectorView2<D>) -> Vector2<D> {
        &self.rot.apply(v) + &self.xy
    }

    fn hat_swap(xi: VectorView2<D>) -> Matrix2x3<D> {
        let mut mat = Matrix2x3::<D>::zeros();
        mat.fixed_view_mut::<2, 1>(0, 0)
            .copy_from(&SO2::hat_swap(xi.as_view()));

        mat.fixed_view_mut::<2, 2>(0, 1)
            .copy_from(&Matrix2::identity());
        mat
    }

    fn to_matrix(&self) -> Matrix3<D> {
        let mut mat = Matrix3::<D>::identity();
        mat.fixed_view_mut::<2, 2>(0, 0)
            .copy_from(&self.rot.to_matrix());
        mat.fixed_view_mut::<2, 1>(0, 2).copy_from(&self.xy);
        mat
    }

    fn from_matrix(mat: MatrixView<3, 3, D>) -> Self {
        let rot = mat.fixed_view::<2, 2>(0, 0).clone_owned();
        let rot = SO2::from_matrix(rot.as_view());

        let xy = mat.fixed_view::<2, 1>(0, 2).into();

        SE2 { rot, xy }
    }
}

impl<D: DualNum> ops::Mul for SE2<D> {
    type Output = SE2<D>;

    fn mul(self, other: Self) -> Self::Output {
        self.compose(&other)
    }
}

impl<D: DualNum> ops::Mul for &SE2<D> {
    type Output = SE2<D>;

    fn mul(self, other: Self) -> Self::Output {
        self.compose(other)
    }
}

impl<D: DualNum> fmt::Display for SE2<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "SE2({:.3}, {:.3}, {:.3})",
            self.rot.log()[0],
            self.xy[0],
            self.xy[1]
        )
    }
}

impl<D: DualNum> fmt::Debug for SE2<D> {
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
