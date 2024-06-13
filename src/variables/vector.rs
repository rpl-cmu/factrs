use crate::linalg::{Vector, VectorX};
use crate::traits::{DualNum, DualVec, Variable};

// ------------------------- Our needs ------------------------- //
impl<const N: usize, D: DualNum> Variable<D> for Vector<N, D> {
    const DIM: usize = N;
    type Dual = Vector<N, DualVec>;

    fn identity() -> Self {
        Vector::zeros()
    }

    fn inverse(&self) -> Self {
        -self
    }

    fn compose(&self, other: &Self) -> Self {
        self + other
    }

    fn exp(delta: &VectorX<D>) -> Self {
        Self::from_iterator(delta.iter().cloned())
    }

    fn log(&self) -> VectorX<D> {
        VectorX::from_iterator(Self::DIM, self.iter().cloned())
    }

    fn dual_self(&self) -> Self::Dual {
        self.map(|x| x.into())
    }
}

impl<D: DualNum> Variable<D> for VectorX<D> {
    const DIM: usize = 0;
    type Dual = VectorX<DualVec>;

    fn identity() -> Self {
        panic!("Cannot create identity for VectorX")
    }

    fn dim(&self) -> usize {
        self.len()
    }

    fn inverse(&self) -> Self {
        -self
    }

    fn compose(&self, other: &Self) -> Self {
        self + other
    }

    fn exp(delta: &VectorX<D>) -> Self {
        delta.clone()
    }

    fn log(&self) -> VectorX<D> {
        self.clone()
    }

    fn dual_self(&self) -> Self::Dual {
        self.map(|x| x.into())
    }
}
