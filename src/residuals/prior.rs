use super::{Residual, Residual1};
use crate::{
    containers::{Symbol, Values},
    dtype,
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
        VectorX,
    },
    variables::{Variable, VariableSafe},
};

#[derive(Clone, Debug, derive_more::Display)]
pub struct PriorResidual<P: Variable> {
    prior: P,
}

impl<P: Variable<Alias<dtype> = P>> PriorResidual<P> {
    pub fn new(prior: P) -> Self {
        Self { prior }
    }
}

impl<P: Variable<Alias<dtype> = P> + VariableSafe + 'static> Residual1 for PriorResidual<P>
where
    AllocatorBuffer<P::Dim>: Sync + Send,
    DefaultAllocator: DualAllocator<P::Dim>,
    DualVector<P::Dim>: Copy,
{
    type Differ = ForwardProp<P::Dim>;
    type V1 = P;
    type DimIn = P::Dim;
    type DimOut = P::Dim;

    fn residual1<D: Numeric>(&self, v: <Self::V1 as Variable>::Alias<D>) -> VectorX<D> {
        Self::V1::dual_convert::<D>(&self.prior).ominus(&v)
    }
}

impl<P: Variable<Alias<dtype> = P> + VariableSafe + 'static> Residual for PriorResidual<P>
where
    AllocatorBuffer<P::Dim>: Sync + Send,
    DefaultAllocator: DualAllocator<P::Dim>,
    DualVector<P::Dim>: Copy,
{
    type DimIn = <Self as Residual1>::DimIn;
    type DimOut = <Self as Residual1>::DimOut;
    type NumVars = Const<1>;
    fn residual(&self, values: &Values, keys: &[Symbol]) -> VectorX {
        self.residual1_values(values, keys)
    }
    fn residual_jacobian(&self, values: &Values, keys: &[Symbol]) -> DiffResult<VectorX, MatrixX> {
        self.residual1_jacobian(values, keys)
    }
}

#[cfg(test)]
mod test {

    use matrixcompare::assert_matrix_eq;

    use super::*;
    use crate::{
        containers::X,
        linalg::{dvector, DefaultAllocator, Diff, DualAllocator, NumericalDiff},
        variables::{VariableSafe, Vector3, SE3, SO3},
    };

    #[cfg(not(feature = "f32"))]
    const PWR: i32 = 6;
    #[cfg(not(feature = "f32"))]
    const TOL: f64 = 1e-6;

    #[cfg(feature = "f32")]
    const PWR: i32 = 3;
    #[cfg(feature = "f32")]
    const TOL: f32 = 1e-3;

    fn test_prior_jacobian<P>(prior: P)
    where
        P: Variable<Alias<dtype> = P> + VariableSafe + 'static,
        AllocatorBuffer<P::Dim>: Sync + Send,
        DefaultAllocator: DualAllocator<P::Dim>,
        DualVector<P::Dim>: Copy,
    {
        let prior_residual = PriorResidual::new(prior);

        let x1 = P::identity();
        let mut values = Values::new();
        values.insert(X(0), x1.clone());
        let jac = prior_residual.residual1_jacobian(&values, &[X(0)]).diff;

        let f = |v: P| {
            let mut vals = Values::new();
            vals.insert(X(0), v.clone());
            Residual1::residual1_values(&prior_residual, &vals, &[X(0)])
        };
        let jac_n = NumericalDiff::<PWR>::jacobian_1(f, &x1).diff;

        eprintln!("jac: {:.3}", jac);
        eprintln!("jac_n: {:.3}", jac_n);

        assert_matrix_eq!(jac, jac_n, comp = abs, tol = TOL);
    }

    #[test]
    fn prior_linear() {
        test_prior_jacobian(Vector3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn prior_so3() {
        let prior = SO3::exp(dvector![0.1, 0.2, 0.3].as_view());
        test_prior_jacobian(prior);
    }

    #[test]
    fn prior_se3() {
        let prior = SE3::exp(dvector![0.1, 0.2, 0.3, 1.0, 2.0, 3.0].as_view());
        test_prior_jacobian(prior);
    }
}
