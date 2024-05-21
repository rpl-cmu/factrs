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

    fn minus(&self, other: &Self) -> Self {
        self - other
    }

    fn plus(&self, other: &Self) -> Self {
        self + other
    }

    fn oplus(&self, delta: &VectorX<D>) -> Self {
        self + delta
    }

    fn ominus(&self, other: &Self) -> VectorX<D> {
        let diff = self.minus(other);
        VectorX::from_iterator(Self::DIM, diff.iter().cloned())
    }

    fn dual_self(&self) -> Self::Dual {
        self.map(|x| x.into())
    }
}
