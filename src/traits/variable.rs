use crate::traits::{DualNum, DualVec};
use crate::variables::VectorD;
use nalgebra as na;
use nalgebra::DMatrix;
use std::fmt::{Debug, Display};
use std::ops::Mul;

pub trait Variable<D: DualNum>: Clone + Sized + Display + Debug {
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

    fn oplus(&self, delta: &VectorD<D>) -> Self;

    fn ominus(&self, other: &Self) -> VectorD<D>;

    fn dual_self(&self) -> Self::Dual;

    fn dual_tangent(&self, idx: usize, total: usize) -> VectorD<DualVec> {
        let mut tv: VectorD<DualVec> = VectorD::zeros(self.dim());
        for (i, tvi) in tv.iter_mut().enumerate() {
            tvi.eps =
                num_dual::Derivative::derivative_generic(na::Dyn(total), na::Const::<1>, idx + i);
        }
        tv
    }

    fn dual(&self, idx: usize, total: usize) -> Self::Dual {
        self.dual_self().oplus(&self.dual_tangent(idx, total))
    }
}

pub trait LieGroup<D: DualNum>: Variable<D> + Mul {
    fn exp(xi: &VectorD<D>) -> Self;

    fn log(&self) -> VectorD<D>;

    fn wedge(xi: &VectorD<D>) -> DMatrix<D>;
}
