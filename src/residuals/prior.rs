use super::{Residual, Residual1};
use crate::containers::Values;
use crate::linalg::{MatrixX, VectorX};
use crate::traits::{DualVec, Key, Variable};

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

    fn residual_jacobian<K: Key>(&self, v: &Values<K, V>, k: &[K]) -> (VectorX, MatrixX) {
        self.residual1_jacobian(v, k)
    }
}

#[cfg(test)]
mod test {

    use super::*;
    use crate::{
        linalg::dvector,
        utils::num_jacobian_11,
        variables::{Symbol, Vector3, SE3, SO3, X},
    };

    #[test]
    fn prior_linear() {
        let prior = Vector3::new(1.0, 2.0, 3.0);
        let prior_residual: PriorResidual<Vector3> = PriorResidual::new(&prior);

        let x1 = Vector3::identity();
        let mut values: Values<Symbol, Vector3> = Values::new();
        values.insert(X(0), x1);
        let (_, jac) = prior_residual.residual1_jacobian(&values, &[X(0)]);

        let f = |v: Vector3| Residual1::<Vector3>::residual1_single(&prior_residual, &v);
        let jac_n = num_jacobian_11(f, x1);

        eprintln!("jac: {:.3}", jac);
        eprintln!("jac_n: {:.3}", jac_n);

        assert!((jac - jac_n).norm() < 1e-4);
    }

    #[test]
    fn prior_so3() {
        let prior = SO3::exp(&dvector![0.1, 0.2, 0.3]);
        let prior_residual = PriorResidual::new(&prior);

        let x1 = SO3::identity();
        let mut values: Values<Symbol, SO3> = Values::new();
        values.insert(X(0), x1.clone());
        let (_, jac) = prior_residual.residual1_jacobian(&values, &[X(0)]);

        let f = |v: SO3| Residual1::<SO3>::residual1_single(&prior_residual, &v);
        let jac_n = num_jacobian_11(f, x1);

        eprintln!("jac: {:.3}", jac);
        eprintln!("jac_n: {:.3}", jac_n);
        assert!((jac - jac_n).norm() < 1e-4);
    }

    #[test]
    fn prior_se3() {
        let prior = SE3::exp(&dvector![0.1, 0.2, 0.3, 1.0, 2.0, 3.0]);

        let prior_residual = PriorResidual::new(&prior);

        let x1 = SE3::identity();
        let mut values: Values<Symbol, SE3> = Values::new();
        values.insert(X(0), x1.clone());
        let (_, jac) = prior_residual.residual1_jacobian(&values, &[X(0)]);

        let f = |v: SE3| Residual1::<SE3>::residual1_single(&prior_residual, &v);
        let jac_n = num_jacobian_11(f, x1);

        eprintln!("jac: {:.3}", jac);
        eprintln!("jac_n: {:.3}", jac_n);

        assert!((jac - jac_n).norm() < 1e-4);
    }
}
