use nalgebra::{allocator::Allocator, DefaultAllocator, DimName};
use num_dual::DualVec;

use crate::{
    dtype,
    linalg::{
        dvector, Const, Derivative, DualAllocator, DualVector, DualVectorGeneric, DualVectorX,
        Matrix1, Matrix2, MatrixView, Numeric, Vector, VectorDim, VectorView1, VectorView2,
        VectorViewX, VectorX,
    },
    variables::{MatrixLieGroup, Variable},
};
use std::{fmt, ops};

use super::{Vector1, Vector2};

#[derive(Clone)]
pub struct SO2<D: Numeric = dtype> {
    a: D,
    b: D,
}

impl<D: Numeric> SO2<D> {
    #[allow(clippy::needless_borrow)]
    pub fn from_theta(theta: D) -> Self {
        SO2 {
            a: (&theta).cos(),
            b: (&theta).sin(),
        }
    }
}

impl<D: Numeric> Variable<D> for SO2<D> {
    type Dim = Const<1>;
    type Alias<DD: Numeric> = SO2<DD>;

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
            a: self.a.clone() * &other.a - self.b.clone() * &other.b,
            b: self.a.clone() * &other.b + self.b.clone() * &other.a,
        }
    }

    fn exp(xi: VectorViewX<D>) -> Self {
        let theta = xi[0].clone();
        SO2::from_theta(theta)
    }

    fn log(&self) -> VectorX<D> {
        dvector![self.b.clone().atan2(self.a.clone())]
    }

    fn dual_convert<DD: Numeric>(other: &Self::Alias<dtype>) -> Self::Alias<DD> {
        Self::Alias::<DD> {
            a: other.a.clone().into(),
            b: other.b.clone().into(),
        }
    }

    fn dual_setup<N: DimName>(idx: usize) -> Self::Alias<DualVectorGeneric<N>>
    where
        <DefaultAllocator as Allocator<dtype, N>>::Buffer: Sync + Send,
        DefaultAllocator: DualAllocator<N>,
    {
        let mut a = DualVectorGeneric::<N>::from_re(1.0);
        a.eps = Derivative::new(Some(VectorDim::<N>::zeros()));

        let mut b = DualVectorGeneric::<N>::from_re(0.0);
        let mut eps = VectorDim::<N>::zeros();
        eps[idx] = 1.0;
        b.eps = Derivative::new(Some(eps));

        SO2 { a, b }
    }
}

impl<D: Numeric> MatrixLieGroup<D> for SO2<D> {
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
            v[0].clone() * &self.a - v[1].clone() * &self.b,
            v[0].clone() * &self.b + v[1].clone() * &self.a,
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

impl<D: Numeric> ops::Mul for SO2<D> {
    type Output = SO2<D>;

    fn mul(self, other: Self) -> Self::Output {
        self.compose(&other)
    }
}

impl<D: Numeric> ops::Mul for &SO2<D> {
    type Output = SO2<D>;

    fn mul(self, other: Self) -> Self::Output {
        self.compose(other)
    }
}

impl<D: Numeric> fmt::Display for SO2<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "SO2({:.3})", self.log()[0])
    }
}

impl<D: Numeric> fmt::Debug for SO2<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "SO2(a: {:.3}, b: {:.3})", self.a, self.b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{test_lie, test_variable};

    test_variable!(SO2);

    test_lie!(SO2);
}
