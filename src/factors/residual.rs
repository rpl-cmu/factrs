use crate::linalg::VectorX;
use crate::traits::{DualVec, Residual, Variable};
use crate::unpack;

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

impl<P: Variable, V: Variable> Residual<V> for PriorResidual<P>
where
    V::Dual: std::convert::TryInto<P::Dual>,
{
    const DIM: usize = P::DIM;

    fn residual(&self, v: &[V::Dual]) -> VectorX<DualVec> {
        let x1: P::Dual = unpack(v[0].clone());
        self.prior.ominus(&x1)
    }
}

// Between Variable
#[derive(Clone, Debug, derive_more::Display)]
pub struct BetweenResidual<P: Variable> {
    delta: P::Dual,
}

impl<P: Variable, V: Variable> Residual<V> for BetweenResidual<P>
where
    V::Dual: TryInto<P::Dual>,
{
    const DIM: usize = P::DIM;

    fn residual(&self, v: &[V::Dual]) -> VectorX<DualVec> {
        let x1: P::Dual = unpack(v[0].clone());
        let x2: P::Dual = unpack(v[1].clone());
        (x1.plus(&self.delta)).ominus(&x2)
    }
}

#[cfg(test)]
mod test {

    use super::*;
    use crate::{
        linalg::{dvector, Matrix3},
        traits::LieGroup,
        variables::{Vector3, SE3, SO3},
    };

    #[test]
    fn prior_linear() {
        let prior = Vector3::new(1.0, 2.0, 3.0);
        let prior_residual: PriorResidual<Vector3> = PriorResidual::new(&prior);

        let x1 = prior;
        let (res, jac) = prior_residual.residual_jacobian(&[x1]);

        eprintln!("res: {:.3}", res);
        eprintln!("jac: {:.3}", jac);

        assert_eq!(res, Vector3::zeros());
        assert_eq!(jac, -Matrix3::identity());
    }

    #[test]
    fn prior_so3() {
        let prior = SO3::exp(&dvector![0.1, 0.2, 0.3]);
        let prior_residual = PriorResidual::new(&prior);

        let x1 = prior.clone();
        let (res, jac) = prior_residual.residual_jacobian(&[x1.clone()]);
        let (res_n, jac_n) = prior_residual.residual_jacobian_numerical(&[x1.clone()]);

        eprintln!("jac: {:.3}", jac);
        eprintln!("jac_n: {:.3}", jac_n);

        assert!((res - res_n).norm() < 1e-4);
        assert!((jac - jac_n).norm() < 1e-4);
    }

    #[test]
    fn prior_se3() {
        let prior = SE3::exp(&dvector![0.1, 0.2, 0.3, 1.0, 2.0, 3.0]);

        let prior_residual = PriorResidual::new(&prior);

        let x1 = prior.clone();
        let (res, jac) = prior_residual.residual_jacobian(&[x1.clone()]);
        let (res_n, jac_n) = prior_residual.residual_jacobian_numerical(&[x1.clone()]);

        eprintln!("jac: {:.3}", jac);
        eprintln!("jac_n: {:.3}", jac_n);

        assert!((res - res_n).norm() < 1e-4);
        assert!((jac - jac_n).norm() < 1e-4);
    }
}
