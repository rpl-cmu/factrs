use crate::dtype;
use crate::linalg::{DualNum, DualVec, Matrix2, Matrix3, MatrixX, Vector2, VectorViewX, VectorX};
use crate::variables::{LieGroup, Variable, SO2};
use std::fmt;
use std::ops;

#[derive(Clone, Debug)]
pub struct SE2<D: DualNum = dtype> {
    rot: SO2<D>,
    xy: Vector2<D>,
}

impl<D: DualNum> SE2<D> {
    pub fn to_matrix(&self) -> Matrix3<D> {
        let mut mat = Matrix3::<D>::identity();
        mat.fixed_view_mut::<2, 2>(0, 0)
            .copy_from(&self.rot.to_matrix());
        mat.fixed_view_mut::<2, 1>(0, 2).copy_from(&self.xy);
        mat
    }

    pub fn from_matrix(mat: &Matrix3<D>) -> Self {
        let rot = mat.fixed_view::<2, 2>(0, 0).into();
        let rot = SO2::from_matrix(&rot);

        let xy = mat.fixed_view::<2, 1>(0, 2).into();

        SE2 { rot, xy }
    }

    pub fn apply(&self, v: &Vector2<D>) -> Vector2<D> {
        self.rot.apply(v) + self.xy.clone()
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
            xy: self.rot.apply(&other.xy) + self.xy.clone(),
        }
    }

    fn inverse(&self) -> Self {
        let inv = self.rot.inverse();
        SE2 {
            xy: -&inv.apply(&self.xy),
            rot: inv,
        }
    }

    #[allow(non_snake_case)]
    fn exp(xi: VectorViewX<D>) -> Self {
        let theta = xi[0].clone();
        let xy = Vector2::new(xi[1].clone(), xi[2].clone());

        let A;
        let B;
        if theta < D::from(1e-5) {
            A = D::from(1.0);
            B = D::from(0.0);
        } else {
            A = theta.clone().sin() / theta.clone();
            B = (D::from(1.0) - theta.clone().cos()) / theta.clone();
        };
        let V = Matrix2::new(A.clone(), -B.clone(), B.clone(), A.clone());

        let rot = SO2::<D>::exp(xi.rows(0, 1));

        SE2 { rot, xy: V * xy }
    }

    #[allow(non_snake_case)]
    fn log(&self) -> VectorX<D> {
        let mut xi = VectorX::zeros(3);
        let theta = self.rot.log()[0].clone();

        let A;
        let B;
        if theta < D::from(1e-5) {
            A = D::from(1.0);
            B = D::from(0.0);
        } else {
            A = theta.clone().sin() / theta.clone();
            B = (D::from(1.0) - theta.clone().cos()) / theta.clone();
        };
        let V = Matrix2::new(A.clone(), -B.clone(), B.clone(), A.clone());

        let Vinv = V.try_inverse().expect("V is not invertible");
        let xy = &Vinv * &self.xy;

        xi[0] = theta;
        xi.as_mut_slice()[1..3].clone_from_slice(xy.as_slice());

        xi
    }

    fn dual_self(&self) -> Self::Dual {
        SE2 {
            rot: self.rot.dual_self(),
            xy: self.xy.dual_self(),
        }
    }
}

impl<D: DualNum> LieGroup<D> for SE2<D> {
    fn hat(xi: VectorViewX<D>) -> MatrixX<D> {
        let mut mat = MatrixX::<D>::zeros(3, 3);
        mat[(0, 1)] = -xi[0].clone();
        mat[(1, 0)] = xi[0].clone();

        mat[(0, 2)] = xi[1].clone();
        mat[(1, 2)] = xi[2].clone();

        mat
    }

    fn adjoint(&self) -> MatrixX<D> {
        let mut mat = MatrixX::<D>::zeros(Self::DIM, Self::DIM);

        let r_mat = self.rot.to_matrix();

        mat.fixed_view_mut::<2, 2>(0, 0).copy_from(&r_mat);
        mat[(0, 2)] = self.xy[2].clone();
        mat[(1, 2)] = -self.xy[1].clone();

        mat
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
        write!(f, "{} {:?}", self.rot, self.xy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::{dvector, Diff, DiffResult, DualNum, ForwardProp, Matrix2x3, Matrix3};
    use matrixcompare::assert_matrix_eq;

    #[test]
    fn matrix() {
        // to_matrix -> from_matrix shoudl give back original vector
        let xi = dvector![0.1, 0.2, 0.3];
        let se2 = SE2::exp(xi.as_view());
        let mat = se2.to_matrix();

        let se2_hat = SE2::from_matrix(&mat);

        assert_matrix_eq!(se2.ominus(&se2_hat), VectorX::zeros(3), comp = float);
    }

    #[test]
    fn jacobian() {
        fn rotate<D: DualNum>(r: SE2<D>) -> VectorX<D> {
            let v = Vector2::new(D::from(1.0), D::from(2.0));
            let rotated = r.apply(&v);
            dvector![rotated[0].clone(), rotated[1].clone()]
        }

        let t = SE2::exp(dvector![0.1, 0.2, 0.3].as_view());
        let DiffResult {
            value: _x,
            diff: dx,
        } = ForwardProp::jacobian_1(rotate, &t);

        let dropper: Matrix2x3 = Matrix2x3::identity();
        let v = dvector!(1.0, 2.0);
        let mut jac = Matrix3::zeros();
        jac[(0, 0)] = -v[1];
        jac[(1, 0)] = v[0];
        jac.fixed_view_mut::<2, 2>(0, 1)
            .copy_from(&Matrix2::identity());

        #[cfg(not(feature = "left"))]
        let dx_exp = dropper * t.to_matrix() * jac;
        // TODO: Verify left jacobian
        #[cfg(feature = "left")]
        let dx_exp = dropper * t.to_matrix() * t.inverse().adjoint() * jac;

        println!("Expected: {}", dx_exp);
        println!("Actual: {}", dx);

        assert_matrix_eq!(dx, dx_exp, comp = float);
    }
}
