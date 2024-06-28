use super::{Residual, Residual2};
use crate::{
    containers::Values,
    impl_residual,
    linalg::{DiffResult, DualVectorX, ForwardProp, MatrixX, VectorX},
    variables::Variable,
};

// Between Variable
#[derive(Clone, Debug, derive_more::Display)]
pub struct BetweenResidual<P: Variable> {
    delta: P::Alias<DualVectorX>,
}

impl<P: Variable> BetweenResidual<P> {
    pub fn new(delta: &P) -> Self {
        Self {
            delta: delta.dual_self(),
        }
    }
}

impl<P: Variable + 'static> Residual2 for BetweenResidual<P> {
    type DimOut = P::Dim;
    type Differ = ForwardProp;
    type V1 = P;
    type V2 = P;

    fn residual2(
        &self,
        v1: P::Alias<DualVectorX>,
        v2: P::Alias<DualVectorX>,
    ) -> VectorX<DualVectorX> {
        v1.compose(&self.delta).ominus(&v2)
    }
}

impl_residual!(2, BetweenResidual<P : Variable>, P);
