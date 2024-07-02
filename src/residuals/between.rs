use nalgebra::{DimNameAdd, DimNameSum};

use super::{Residual, Residual2};
use crate::{
    containers::{Symbol, Values},
    dtype,
    impl_safe_residual,
    linalg::{
        AllocatorBuffer,
        Const,
        DefaultAllocator,
        DiffResult,
        DualAllocator,
        DualVector,
        ForwardProp,
        MatrixX,
        Numeric,
        Vector1,
        Vector2,
        Vector3,
        Vector4,
        Vector5,
        Vector6,
        VectorX,
    },
    variables::{Variable, VariableSafe, SE2, SE3, SO2, SO3},
};

impl_safe_residual!(
    BetweenResidual<Vector1>,
    BetweenResidual<Vector2>,
    BetweenResidual<Vector3>,
    BetweenResidual<Vector4>,
    BetweenResidual<Vector5>,
    BetweenResidual<Vector6>,
    BetweenResidual<SE2>,
    BetweenResidual<SE3>,
    BetweenResidual<SO2>,
    BetweenResidual<SO3>,
);

// Between Variable
#[derive(Clone, Debug, derive_more::Display)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct BetweenResidual<P: Variable> {
    delta: P,
}

impl<P: Variable> BetweenResidual<P> {
    pub fn new(delta: P) -> Self {
        Self { delta }
    }
}

impl<P: Variable<Alias<dtype> = P> + VariableSafe + 'static> Residual2 for BetweenResidual<P>
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

    fn residual2<D: Numeric>(&self, v1: P::Alias<D>, v2: P::Alias<D>) -> VectorX<D> {
        let delta = P::dual_convert::<D>(&self.delta);
        v1.compose(&delta).ominus(&v2)
    }
}

impl<P> Residual for BetweenResidual<P>
where
    AllocatorBuffer<DimNameSum<P::Dim, P::Dim>>: Sync + Send,
    DefaultAllocator: DualAllocator<DimNameSum<P::Dim, P::Dim>>,
    DualVector<DimNameSum<P::Dim, P::Dim>>: Copy,
    P: Variable<Alias<dtype> = P> + VariableSafe + 'static,
    P::Dim: DimNameAdd<P::Dim>,
{
    type DimOut = <Self as Residual2>::DimOut;
    type DimIn = <Self as Residual2>::DimIn;
    type NumVars = Const<2>;

    fn residual(&self, values: &Values, keys: &[Symbol]) -> VectorX {
        self.residual2_values(values, keys)
    }

    fn residual_jacobian(&self, values: &Values, keys: &[Symbol]) -> DiffResult<VectorX, MatrixX> {
        self.residual2_jacobian(values, keys)
    }
}
