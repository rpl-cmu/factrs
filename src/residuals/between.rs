use core::fmt;

use nalgebra::{DimNameAdd, DimNameSum};

use super::Residual2;
#[allow(unused_imports)]
use crate::{
    containers::{Key, Values},
    linalg::{
        AllocatorBuffer, Const, DefaultAllocator, DiffResult, DimName, DualAllocator, DualVector,
        ForwardProp, MatrixX, Numeric, VectorX,
    },
    variables::{
        Variable, VariableUmbrella, VectorVar1, VectorVar2, VectorVar3, VectorVar4, VectorVar5,
        VectorVar6, SE2, SE3, SO2, SO3,
    },
};

/// Binary factor between variables.
///
/// This residual is used to enforce a constraint between two variables.
/// Specifically it computes
///
/// $$
/// r = (v_1 z) \ominus v_2
/// $$
///
/// where $z$ is the measured value.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct BetweenResidual<P: Variable> {
    delta: P,
}

impl<P: Variable> BetweenResidual<P> {
    pub fn new(delta: P) -> Self {
        Self { delta }
    }
}

#[factrs::tag]
impl<P: VariableUmbrella + 'static> Residual2 for BetweenResidual<P>
where
    AllocatorBuffer<DimNameSum<P::Dim, P::Dim>>: Sync + Send,
    DefaultAllocator: DualAllocator<DimNameSum<P::Dim, P::Dim>>,
    DualVector<DimNameSum<P::Dim, P::Dim>>: Copy,
    P::Dim: DimNameAdd<P::Dim>,
{
    type Differ = ForwardProp<DimNameSum<P::Dim, P::Dim>>;
    type V1 = P;
    type V2 = P;
    type DimOut = P::Dim;
    type DimIn = DimNameSum<P::Dim, P::Dim>;

    fn residual2<T: Numeric>(&self, v1: P::Alias<T>, v2: P::Alias<T>) -> VectorX<T> {
        let delta = P::dual_convert::<T>(&self.delta);
        v1.compose(&delta).ominus(&v2)
    }
}

impl<P: Variable> fmt::Display for BetweenResidual<P> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}
