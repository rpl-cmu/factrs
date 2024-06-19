use crate::linalg::{DualNum, DualVec, Vector, VectorViewX, VectorX};
use crate::variables::Variable;

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

    fn exp(delta: VectorViewX<D>) -> Self {
        Self::from_iterator(delta.iter().cloned())
    }

    fn log(&self) -> VectorX<D> {
        VectorX::from_iterator(Self::DIM, self.iter().cloned())
    }

    fn dual_self(&self) -> Self::Dual {
        self.map(|x| x.into())
    }
}
