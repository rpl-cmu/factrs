use crate::{
    containers::{GraphGeneric, Key, Values},
    linear::LinearSolver,
    noise::NoiseModel,
    residuals::Residual,
    robust::RobustCost,
    variables::Variable,
};

pub struct OptimizerParams<S: LinearSolver> {
    pub max_iterations: usize,
    pub error_tol_relative: f64,
    pub error_tol_absolute: f64,
    pub error_tol: f64,
    pub solver: S,
}

pub trait Optimizer {
    fn iterate<K: Key, V: Variable, R: Residual<V>, N: NoiseModel, C: RobustCost, S: LinearSolver>(
        params: &mut OptimizerParams<S>,
        graph: &GraphGeneric<K, V, R, N, C>,
        values: &mut Values<K, V>,
    );

    fn optimize<
        K: Key,
        V: Variable,
        R: Residual<V>,
        N: NoiseModel,
        C: RobustCost,
        S: LinearSolver,
    >(
        params: OptimizerParams<S>,
        graph: &GraphGeneric<K, V, R, N, C>,
        values: &mut Values<K, V>,
    ) {
        let mut params = params;

        // Check if we need to optimize at all
        let error_old = graph.error(values);
        if error_old <= params.error_tol {
            return;
        }

        // Begin iterations
        let error_new = error_old;
        for _ in 0..params.max_iterations {
            let error_old = error_new;
            Self::iterate(&mut params, graph, values);

            // Evaulate error again to see how we did
            let error_new = graph.error(values);

            // Check if we need to stop
            if error_new <= params.error_tol {
                return;
            }
            let error_decrease_abs = error_old - error_new;
            if error_decrease_abs <= params.error_tol_absolute {
                return;
            }
            let error_decrease_rel = error_decrease_abs / error_old;
            if error_decrease_rel <= params.error_tol_relative {
                return;
            }
        }
    }
}

mod gauss_newton;
pub use gauss_newton::GaussNewton;

#[cfg(test)]
mod test {
    use faer::assert_matrix_eq;

    use crate::{
        containers::{Symbol, X},
        factors::FactorGeneric,
        linalg::Vector3,
        linear::CholeskySolver,
        make_enum_residual,
        noise::GaussianNoise,
        residuals::{BetweenResidual, PriorResidual},
        robust::L2,
    };

    use super::*;

    #[test]
    fn test_prior() {
        let params = OptimizerParams {
            max_iterations: 100,
            error_tol_relative: 1e-6,
            error_tol_absolute: 1e-6,
            error_tol: 1e-6,
            solver: CholeskySolver::default(),
        };

        let p = Vector3::new(3.0, 2.0, 1.0);
        let mut values = Values::new();
        values.insert(X(0), Vector3::new(1.0, 2.0, 3.0));

        let mut graph: GraphGeneric<Symbol, Vector3, PriorResidual<Vector3>, GaussianNoise, L2> =
            GraphGeneric::new();
        let res = PriorResidual::new(&p);
        let factor = FactorGeneric::new(vec![X(0)], res).build();
        graph.add_factor(factor);

        GaussNewton::optimize(params, &graph, &mut values);

        let out = values.get(&X(0)).unwrap();
        assert_matrix_eq!(out, p, comp = abs, tol = 1e-6);
    }

    #[test]
    fn test_between() {
        let params = OptimizerParams {
            max_iterations: 100,
            error_tol_relative: 1e-6,
            error_tol_absolute: 1e-6,
            error_tol: 1e-6,
            solver: CholeskySolver::default(),
        };

        let p1 = Vector3::new(3.0, 2.0, 1.0);
        let p2 = Vector3::new(1.0, 2.0, 3.0);
        let mut values = Values::new();
        values.insert(X(0), Vector3::zeros());
        values.insert(X(1), Vector3::zeros());

        make_enum_residual!(
            MyResiduals,
            Vector3,
            PriorResidual<Vector3>,
            BetweenResidual<Vector3>
        );

        let mut graph: GraphGeneric<Symbol, Vector3, MyResiduals, GaussianNoise, L2> =
            GraphGeneric::new();
        let res = PriorResidual::new(&p1);
        let factor = FactorGeneric::new(vec![X(0)], res).build();
        graph.add_factor(factor);

        let res = BetweenResidual::new(&(p2 - p1));
        let factor = FactorGeneric::new(vec![X(0), X(1)], res).build();
        graph.add_factor(factor);

        GaussNewton::optimize(params, &graph, &mut values);

        let out1 = values.get(&X(0)).unwrap();
        println!("Got {:?}, expected {:?}", out1, p1);
        assert_matrix_eq!(out1, p1, comp = abs, tol = 1e-6);

        let out2 = values.get(&X(1)).unwrap();
        println!("Got {:?}, expected {:?}", out2, p2);
        assert_matrix_eq!(out2, p2, comp = abs, tol = 1e-6);
    }
}
