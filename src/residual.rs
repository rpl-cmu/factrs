use crate::linalg::VectorX;
use crate::traits::{DualVec, Residual, Variable};
use crate::{dtype, unpack};

// Prior Variable
pub struct PriorResidual<P: Variable<dtype>> {
    prior: P::Dual,
}

impl<P: Variable<dtype>> PriorResidual<P> {
    pub fn new(prior: &P) -> Self {
        Self {
            prior: prior.dual_self(),
        }
    }
}

impl<P: Variable<dtype>, V: Variable<dtype>> Residual<V> for PriorResidual<P>
where
    V::Dual: TryInto<P::Dual>,
{
    const DIM: usize = V::DIM;

    fn residual(&self, v: &[V::Dual]) -> VectorX<DualVec> {
        let x1: P::Dual = unpack(v[0].clone());
        self.prior.ominus(&x1)
    }
}

// Between Variable
pub struct BetweenResidual<P: Variable<dtype>> {
    delta: P::Dual,
}

impl<P: Variable<dtype>, V: Variable<dtype>> Residual<V> for BetweenResidual<P>
where
    V::Dual: TryInto<P::Dual>,
{
    const DIM: usize = V::DIM;

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
        variables::{Vector3, Vector6, SE3, SO3},
    };

    #[test]
    fn prior_linear() {
        let prior = Vector3::new(1.0, 2.0, 3.0);
        let prior_residual: PriorResidual<Vector3> = PriorResidual::new(&prior);

        let x1 = prior;
        let (res, jac) = prior_residual.residual_jacobian(&[x1]);

        println!("res: {:}", res);
        println!("jac: {:}", jac);

        assert_eq!(res, Vector3::zeros());
        assert_eq!(jac, -Matrix3::identity());
    }

    #[test]
    fn prior_so3() {
        // It would be preferable to test this when prior != x0
        // But I don't believe the jacobian actually has a closed form solution
        let prior = SO3::exp(&dvector![0.1, 0.2, 0.3]);
        let prior_residual = PriorResidual::new(&prior);

        let x1 = prior.clone();
        let (res, jac) = prior_residual.residual_jacobian(&[x1.clone()]);

        println!("res: {:}", res);
        println!("jac: {:}", jac);

        // This is a first order approximation of the jacobian
        let jac_exp = -prior.minus(&x1).adjoint();

        assert!((res - Vector3::zeros()).norm() < 1e-4);
        assert!((jac - jac_exp).norm() < 1e-4);
    }

    #[test]
    fn prior_se3() {
        // TODO: This still seems off for SE3.
        // When rotations are included, the bottom left of Jacobian is off by negative
        // When they aren't included, the bottom left of Jacobian is off by factor of 2 and negative

        // It would be preferable to test this when prior != x0
        // But I don't believe the derivative actually has a closed form solution
        let prior = SE3::exp(&dvector![1e-3, 0.0, 0.0, 1.0, 2.0, 3.0]);

        let prior_residual = PriorResidual::new(&prior);

        let x1 = SE3::identity();
        let (res, jac) = prior_residual.residual_jacobian(&[x1.clone()]);

        // This is a first order approximation of the jacobian
        let jac_exp = -prior.minus(&x1).adjoint();

        println!("res: {:}", res);
        println!("jac: {:}", jac);
        println!("jac_exp: {:}", jac_exp);

        assert!((res - Vector6::zeros()).norm() < 1e-4);
        assert!((jac - jac_exp).norm() < 1e-4);
    }
}
