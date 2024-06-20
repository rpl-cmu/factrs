use super::{Residual, Residual2};
use crate::containers::{Key, Values};
use crate::linalg::{DiffResult, DualVec, ForwardProp, MatrixX, VectorX};
use crate::variables::Variable;

// // Between Variable
#[derive(Clone, Debug, derive_more::Display)]
pub struct BetweenResidual<P: Variable> {
    delta: P::Dual,
}

impl<P: Variable> BetweenResidual<P> {
    pub fn new(delta: &P) -> Self {
        Self {
            delta: delta.dual_self(),
        }
    }
}

impl<P: Variable, V: Variable> Residual2<V> for BetweenResidual<P>
where
    for<'a> &'a V: std::convert::TryInto<&'a P>,
{
    const DIM: usize = P::DIM;
    type Differ = ForwardProp;
    type V1 = P;
    type V2 = P;

    fn residual2(&self, v1: P::Dual, v2: P::Dual) -> VectorX<DualVec> {
        v1.compose(&self.delta).ominus(&v2)
    }
}

impl<V: Variable, P: Variable> Residual<V> for BetweenResidual<P>
where
    for<'a> &'a V: std::convert::TryInto<&'a P>,
{
    const DIM: usize = <Self as Residual2<V>>::DIM;

    fn residual_jacobian<K: Key>(&self, v: &Values<K, V>, k: &[K]) -> DiffResult<VectorX, MatrixX> {
        self.residual2_jacobian(v, k)
    }
}
