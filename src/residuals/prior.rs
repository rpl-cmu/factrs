use super::{Residual, Residual1};
use crate::containers::{Key, Values};
use crate::linalg::{DiffResult, DualVec, ForwardProp, MatrixX, VectorX};
use crate::variables::Variable;

#[derive(Clone, Debug, derive_more::Display)]
pub struct PriorResidual<P: Variable> {
    prior: P::Dual,
}

impl<P: Variable> PriorResidual<P> {
    pub fn new(prior: &P) -> Self {
        Self {
            prior: prior.dual_self(),
        }
    }
}

impl<P: Variable, V: Variable> Residual1<V> for PriorResidual<P>
where
    for<'a> &'a V: std::convert::TryInto<&'a P>,
{
    const DIM: usize = P::DIM;
    type V1 = P;
    type Differ = ForwardProp;

    fn residual1(&self, v: P::Dual) -> VectorX<DualVec> {
        self.prior.ominus(&v)
    }
}

// TODO: This is mostly boilerplate, can we generate this?
// Can't use blanket implementation for Residual b/c there'll be potential overlap with Residual2, Residual3, etc.
impl<V: Variable, P: Variable> Residual<V> for PriorResidual<P>
where
    for<'a> &'a V: std::convert::TryInto<&'a P>,
{
    const DIM: usize = <PriorResidual<P> as Residual1<V>>::DIM;

    fn residual_jacobian<K: Key>(&self, v: &Values<K, V>, k: &[K]) -> DiffResult<VectorX, MatrixX> {
        self.residual1_jacobian(v, k)
    }
}

#[cfg(test)]
mod test {

    use super::*;
    use crate::{
        containers::{Symbol, X},
        linalg::dvector,
        linalg::NumericalDiff,
        variables::{Vector3, SE3, SO3},
    };
    use matrixcompare::assert_matrix_eq;

    fn test_prior_jacobian<P: Variable>(prior: P) {
        let prior_residual = PriorResidual::new(&prior);

        let x1 = P::identity();
        let mut values: Values<Symbol, P> = Values::new();
        values.insert(X(0), x1.clone());
        let jac = prior_residual.residual1_jacobian(&values, &[X(0)]).diff;

        let f = |v: P| Residual1::<P>::residual1_single(&prior_residual, &v);
        let jac_n = NumericalDiff::<6>::jacobian_1(f, &x1).diff;

        eprintln!("jac: {:.3}", jac);
        eprintln!("jac_n: {:.3}", jac_n);

        assert_matrix_eq!(jac, jac_n, comp = abs, tol = 1e-6);
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
