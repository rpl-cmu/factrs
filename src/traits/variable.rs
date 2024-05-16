use crate::dtype;
use crate::traits::DualNum;
use crate::variables::VectorD;
use nalgebra::DMatrix;
use std::fmt::{Debug, Display};
use std::ops::Mul;

pub trait Variable<D: DualNum<dtype>>: Clone + Sized + Display + Debug {
    const DIM: usize;

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
}

pub trait LieGroup<D: DualNum<dtype>>: Variable<D> + Mul {
    fn exp(xi: &VectorD<D>) -> Self;

    fn log(&self) -> VectorD<D>;

    fn wedge(xi: &VectorD<D>) -> DMatrix<D>;
}
