use super::{Residual, Residual2};
use crate::linalg::{MatrixX, VectorX};
use crate::traits::{DualVec, Key, Variable};
use crate::variables::Values;

// // Between Variable
#[derive(Clone, Debug, derive_more::Display)]
pub struct BetweenResidual<P: Variable> {
    delta: P::Dual,
}

impl<P: Variable, V: Variable> Residual2<V> for BetweenResidual<P>
where
    for<'a> &'a V: std::convert::TryInto<&'a P>,
{
    const DIM: usize = P::DIM;
    type V1 = P;
    type V2 = P;

    fn residual2(&self, v1: P::Dual, v2: P::Dual) -> VectorX<DualVec> {
        (v1.compose(&self.delta)).ominus(&v2)
    }
}

impl<V: Variable, P: Variable> Residual<V> for BetweenResidual<P>
where
    for<'a> &'a V: std::convert::TryInto<&'a P>,
{
    const DIM: usize = 0;

    fn residual_jacobian<K: Key>(&self, _: &Values<K, V>, _: &[K]) -> (VectorX, MatrixX) {
        (VectorX::zeros(0), MatrixX::zeros(0, 0))
    }
}
