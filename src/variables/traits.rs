use crate::dtype;
use crate::linalg::{Const, DualNum, DualVec, Dyn, MatrixX, VectorX};

use std::fmt::{Debug, Display};

pub trait Variable<D: DualNum = dtype>: Clone + Sized + Display + Debug {
    const DIM: usize;
    type Dual: Variable<DualVec>;

    // Group operations
    fn identity() -> Self;
    fn inverse(&self) -> Self;
    fn compose(&self, other: &Self) -> Self;
    // Make exp/log consume their arguments?
    fn exp(delta: &VectorX<D>) -> Self; // trivial if linear (just itself)
    fn log(&self) -> VectorX<D>; // trivial if linear (just itself)

    // Conversion to dual space
    fn dual_self(&self) -> Self::Dual;

    // Helpers for enum
    fn dim(&self) -> usize {
        Self::DIM
    }
    fn identity_enum(&self) -> Self {
        Self::identity()
    }

    // Moves to and from vector space
    fn oplus(&self, delta: &VectorX<D>) -> Self {
        if cfg!(feature = "left") {
            Self::exp(delta).compose(self)
        } else {
            self.compose(&Self::exp(delta))
        }
    }
    fn ominus(&self, other: &Self) -> VectorX<D> {
        if cfg!(feature = "left") {
            (self.compose(&other.inverse())).log()
        } else {
            (other.inverse().compose(self)).log()
        }
    }

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

pub trait LieGroup<D: DualNum>: Variable<D> {
    fn hat(xi: &VectorX<D>) -> MatrixX<D>;

    fn adjoint(&self) -> MatrixX<D>;
}
