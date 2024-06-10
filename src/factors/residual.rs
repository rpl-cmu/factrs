use crate::linalg::{MatrixX, VectorX};
use crate::traits::{DualVec, Key, Residual, Residual1, Residual2, Variable};
use crate::variables::Values;

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

impl<V: Variable, P: Variable> Residual<V> for PriorResidual<P>
where
    for<'a> &'a V: std::convert::TryInto<&'a P>,
{
    const DIM: usize = <PriorResidual<P> as Residual1<V>>::DIM;

    fn residual_jacobian<K: Key>(&self, v: &Values<K, V>, k: &[K]) -> (VectorX, MatrixX) {
        <Self as Residual1<V>>::residual_jacobian(&self, v, k)
    }
}

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

#[cfg(test)]
mod test {

    // use super::*;
    // use crate::{
    //     linalg::{dvector, Matrix3},
    //     variables::{Vector3, SE3, SO3},
    // };

    // #[test]
    // fn prior_linear() {
    //     let prior = Vector3::new(1.0, 2.0, 3.0);
    //     let prior_residual: PriorResidual<Vector3> = PriorResidual::new(&prior);

    //     let x1 = prior;
    //     let (res, jac) = prior_residual.residual_jacobian(&[x1]);

    //     eprintln!("res: {:.3}", res);
    //     eprintln!("jac: {:.3}", jac);

    //     assert_eq!(res, Vector3::zeros());
    //     assert_eq!(jac, -Matrix3::identity());
    // }

    // #[test]
    // fn prior_so3() {
    //     let prior = SO3::exp(&dvector![0.1, 0.2, 0.3]);
    //     let prior_residual = PriorResidual::new(&prior);

    //     let x1 = prior.clone();
    //     let (res, jac) = prior_residual.residual_jacobian(&[x1.clone()]);
    //     let (res_n, jac_n) = prior_residual.residual_jacobian_numerical(&[x1.clone()]);

    //     eprintln!("jac: {:.3}", jac);
    //     eprintln!("jac_n: {:.3}", jac_n);

    //     assert!((res - res_n).norm() < 1e-4);
    //     assert!((jac - jac_n).norm() < 1e-4);
    // }

    // #[test]
    // fn prior_se3() {
    //     let prior = SE3::exp(&dvector![0.1, 0.2, 0.3, 1.0, 2.0, 3.0]);

    //     let prior_residual = PriorResidual::new(&prior);

    //     let x1 = prior.clone();
    //     let (res, jac) = prior_residual.residual_jacobian(&[x1.clone()]);
    //     let (res_n, jac_n) = prior_residual.residual_jacobian_numerical(&[x1.clone()]);

    //     eprintln!("jac: {:.3}", jac);
    //     eprintln!("jac_n: {:.3}", jac_n);

    //     assert!((res - res_n).norm() < 1e-4);
    //     assert!((jac - jac_n).norm() < 1e-4);
    // }
}
