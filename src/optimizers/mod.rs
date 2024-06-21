use crate::{
    containers::{GraphGeneric, Key, Values},
    dtype,
    linear::{CholeskySolver, LinearSolver},
    noise::NoiseModel,
    residuals::Residual,
    robust::RobustCost,
    variables::Variable,
};

pub struct OptimizerParams<S: LinearSolver = CholeskySolver> {
    pub max_iterations: usize,
    pub error_tol_relative: dtype,
    pub error_tol_absolute: dtype,
    pub error_tol: dtype,
    pub solver: S,
}

impl<S: LinearSolver> Default for OptimizerParams<S> {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            error_tol_relative: 1e-6,
            error_tol_absolute: 1e-6,
            error_tol: 0.0,
            solver: S::default(),
        }
    }
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
        let mut error_old = graph.error(values);
        if error_old <= params.error_tol {
            log::info!("Error is already below tolerance, skipping optimization");
            return;
        }

        log::info!(
            "{:^5} | {:^12} | {:^12} | {:^12}",
            "Iter",
            "Error",
            "ErrorAbs",
            "ErrorRel"
        );
        log::info!(
            "{:^5} | {:^12} | {:^12} | {:^12}",
            "-----",
            "------------",
            "------------",
            "------------"
        );
        log::info!(
            "{:^5} | {:^12.4e} | {:^12} | {:^12}",
            0,
            error_old,
            "-",
            "-"
        );

        // Begin iterations
        let mut error_new = error_old;
        for i in 1..params.max_iterations + 1 {
            error_old = error_new;
            Self::iterate(&mut params, graph, values);

            // Evaulate error again to see how we did
            error_new = graph.error(values);

            // Check if we need to stop
            if error_new <= params.error_tol {
                log::info!("Error is below tolerance, stopping optimization");
                return;
            }
            let error_decrease_abs = error_old - error_new;
            if error_decrease_abs <= params.error_tol_absolute {
                log::info!("Error decrease is below absolute tolerance, stopping optimization");
                return;
            }
            let error_decrease_rel = error_decrease_abs / error_old;
            if error_decrease_rel <= params.error_tol_relative {
                log::info!("Error decrease is below relative tolerance, stopping optimization");
                return;
            }

            log::info!(
                "{:^5} | {:^12.4e} | {:^12.4e} | {:^12.4e}",
                i,
                error_new,
                error_decrease_abs,
                error_decrease_rel
            )
        }
    }
}

mod gauss_newton;
pub use gauss_newton::GaussNewton;

#[cfg(test)]
mod test {
    use faer::assert_matrix_eq;
    use nalgebra::dvector;

    use crate::{
        bundle::DefaultBundle,
        containers::{Graph, X},
        factors::FactorGeneric,
        linalg::{Vector3, VectorX},
        residuals::{BetweenResidual, PriorResidual, ResidualEnum},
        variables::{VariableEnum, SE3, SO3},
    };

    use super::*;

    fn prior<T>(p: T)
    where
        T: Variable + Into<VariableEnum>,
        for<'a> &'a VariableEnum: TryInto<&'a T>,
        PriorResidual<T>: Into<ResidualEnum>,
    {
        let mut values = Values::new();
        values.insert(X(0), T::identity());

        let mut graph: Graph<DefaultBundle> = GraphGeneric::new();
        let res = PriorResidual::new(&p);
        let factor = FactorGeneric::new(vec![X(0)], res).build();
        graph.add_factor(factor);

        let params: OptimizerParams = OptimizerParams::default();
        GaussNewton::optimize(params, &graph, &mut values);

        let out: &T = values.get(&X(0)).unwrap().try_into().ok().unwrap();
        assert_matrix_eq!(
            out.ominus(&p),
            VectorX::zeros(T::DIM),
            comp = abs,
            tol = 1e-6
        );
    }

    fn between<T>(p1: T, p2: T)
    where
        T: Variable + Into<VariableEnum>,
        for<'a> &'a VariableEnum: TryInto<&'a T>,
        PriorResidual<T>: Into<ResidualEnum>,
        BetweenResidual<T>: Into<ResidualEnum>,
    {
        let mut values = Values::new();
        values.insert(X(0), T::identity());
        values.insert(X(1), T::identity());

        let mut graph: Graph<DefaultBundle> = GraphGeneric::new();
        let res = PriorResidual::new(&p1);
        let factor = FactorGeneric::new(vec![X(0)], res).build();
        graph.add_factor(factor);

        let diff = p2.minus(&p1);
        let res = BetweenResidual::new(&diff);
        let factor = FactorGeneric::new(vec![X(0), X(1)], res).build();
        graph.add_factor(factor);

        let params: OptimizerParams = OptimizerParams::default();
        GaussNewton::optimize(params, &graph, &mut values);

        let out1 = values.get(&X(0)).unwrap().try_into().ok().unwrap();
        assert_matrix_eq!(
            out1.ominus(&p1),
            VectorX::zeros(T::DIM),
            comp = abs,
            tol = 1e-6
        );

        let out2 = values.get(&X(1)).unwrap().try_into().ok().unwrap();
        assert_matrix_eq!(
            out2.ominus(&p2),
            VectorX::zeros(T::DIM),
            comp = abs,
            tol = 1e-6
        );
    }

    #[test]
    fn priorvector3() {
        let p = Vector3::new(1.0, 2.0, 3.0);
        prior(p);
    }

    #[test]
    fn priorso3() {
        let p = SO3::exp(dvector![0.1, 0.2, 0.3].as_view());
        prior(p);
    }

    #[test]
    fn priorse3() {
        pretty_env_logger::init();
        let p = SE3::exp(dvector![0.1, 0.2, 0.3, 0.4, 0.5, 0.6].as_view());
        prior(p);
    }

    #[test]
    fn betweenvector3() {
        let p1 = Vector3::new(3.0, 2.0, 1.0);
        let p2 = Vector3::new(1.0, 2.0, 3.0);
        between(p1, p2);
    }

    #[test]
    fn betweenso3() {
        let p1 = SO3::exp(dvector![0.1, 0.2, 0.3].as_view());
        let p2 = SO3::exp(dvector![0.3, 0.2, 0.1].as_view());
        between(p1, p2);
    }

    #[test]
    fn betweense3() {
        let p1 = SE3::exp(dvector![0.1, 0.2, 0.3, 0.4, 0.5, 0.6].as_view());
        let p2 = SE3::exp(dvector![0.3, 0.2, 0.1, 0.6, 0.5, 0.4].as_view());
        between(p1, p2);
    }
}
