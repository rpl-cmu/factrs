use super::{Residual, Residual1};
use crate::{
    containers::Values,
    impl_residual,
    linalg::{DiffResult, DualVectorX, ForwardProp, MatrixX, VectorX},
    variables::Variable,
};

#[derive(Clone, Debug, derive_more::Display)]
pub struct PriorResidual<P: Variable> {
    prior: P::Alias<DualVectorX>,
}

impl<P: Variable> PriorResidual<P> {
    pub fn new(prior: &P) -> Self {
        Self {
            prior: prior.dual_self(),
        }
    }
}

impl<P: Variable + 'static> Residual1 for PriorResidual<P> {
    type DimOut = P::Dim;
    type V1 = P;
    type Differ = ForwardProp;

    fn residual1(&self, v: <Self::V1 as Variable>::Alias<DualVectorX>) -> VectorX<DualVectorX> {
        self.prior.ominus(&v)
    }
}

impl_residual!(1, PriorResidual<P : Variable>, P);

#[cfg(test)]
mod test {

    use super::*;
    use crate::{
        containers::X,
        linalg::{dvector, NumericalDiff},
        variables::{Vector3, SE3, SO3},
    };
    use matrixcompare::assert_matrix_eq;

    #[cfg(not(feature = "f32"))]
    const PWR: i32 = 6;
    #[cfg(not(feature = "f32"))]
    const TOL: f64 = 1e-6;

    #[cfg(feature = "f32")]
    const PWR: i32 = 3;
    #[cfg(feature = "f32")]
    const TOL: f32 = 1e-3;

    fn test_prior_jacobian<P: Variable + 'static>(prior: P) {
        let prior_residual = PriorResidual::new(&prior);

        let x1 = P::identity();
        let mut values = Values::new();
        values.insert(X(0), x1.clone());
        let jac = prior_residual.residual1_jacobian(&values, &[X(0)]).diff;

        let f = |v: P| {
            let mut vals = Values::new();
            vals.insert(X(0), v.clone());
            Residual1::residual1_single(&prior_residual, &vals, &[X(0)])
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
