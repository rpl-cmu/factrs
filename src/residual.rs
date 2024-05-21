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
        variables::{Vector3, SO3},
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
        assert_eq!(jac, Matrix3::identity());
    }

    #[test]
    fn prior_so3() {
        let prior = SO3::exp(&dvector![0.1, 0.2, 0.3]);
        let prior_residual = PriorResidual::new(&prior);

        let x1 = SO3::identity();
        let (res, jac) = prior_residual.residual_jacobian(&[x1]);

        println!("res: {:}", res);
        println!("jac: {:}", jac);

        assert_eq!(res, Vector3::zeros());
        assert_eq!(jac, Matrix3::identity());
    }
}
