use crate::dtype;
use crate::linalg::{Const, Dyn, MatrixX, VectorX};
use crate::traits::{DualNum, DualVec};

use std::fmt::{Debug, Display};
use std::ops::Mul;

pub trait Variable<D: DualNum = dtype>: Clone + Sized + Display + Debug {
    const DIM: usize;
    type Dual: Variable<DualVec>;

    fn dim(&self) -> usize {
        Self::DIM
    }
    fn identity() -> Self;
    fn identity_enum(&self) -> Self {
        Self::identity()
    }
    fn inverse(&self) -> Self;

    // Stays in the manifold
    fn minus(&self, other: &Self) -> Self;
    fn plus(&self, other: &Self) -> Self;

    // Moves into vector space
    fn oplus(&self, delta: &VectorX<D>) -> Self;
    fn ominus(&self, other: &Self) -> VectorX<D>;

    // Conversion to dual space
    fn dual_self(&self) -> Self::Dual;

    // Create tangent vector w/ duals set up properly
    fn dual_tangent(&self, idx: usize, total: usize) -> VectorX<DualVec> {
        let mut tv: VectorX<DualVec> = VectorX::zeros(self.dim());
        for (i, tvi) in tv.iter_mut().enumerate() {
            tvi.eps = num_dual::Derivative::derivative_generic(Dyn(total), Const::<1>, idx + i);
        }
        tv
    }
    // Applies the tangent vector in dual space
    fn dual(&self, idx: usize, total: usize) -> Self::Dual {
        self.dual_self().oplus(&self.dual_tangent(idx, total))
    }
}

pub trait LieGroup<D: DualNum>: Variable<D> + Mul {
    fn exp(xi: &VectorX<D>) -> Self;

    fn log(&self) -> VectorX<D>;

    fn hat(xi: &VectorX<D>) -> MatrixX<D>;

    fn adjoint(&self) -> MatrixX<D>;
}
