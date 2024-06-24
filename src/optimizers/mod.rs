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

mod macros;

mod gauss_newton;
pub use gauss_newton::GaussNewton;

mod levenberg_marquardt;
pub use levenberg_marquardt::LevenMarquardt;

// These aren't tests themselves, but are helpers to test optimizers
#[cfg(test)]
pub mod test {
    use faer::assert_matrix_eq;

    use crate::{
        bundle::DefaultBundle,
        containers::{Graph, X},
        factors::FactorGeneric,
        linalg::VectorX,
        residuals::{BetweenResidual, PriorResidual, ResidualEnum},
        variables::VariableEnum,
    };

    use super::*;

    pub fn optimize_prior<O, T>()
    where
        T: Variable + Into<VariableEnum>,
        for<'a> &'a VariableEnum: TryInto<&'a T>,
        PriorResidual<T>: Into<ResidualEnum>,
        O: Optimizer,
    {
        let t = VectorX::from_fn(T::DIM, |_, i| ((i + 1) as dtype) / 10.0);
        let p = T::exp(t.as_view());

        let mut values = Values::new();
        values.insert(X(0), T::identity());

        let mut graph = Graph::<DefaultBundle>::new();
        let res = PriorResidual::new(&p);
        let factor = FactorGeneric::new(vec![X(0)], res).build();
        graph.add_factor(factor);

        let params: OptimizerParams = OptimizerParams::default();
        O::optimize(params, &graph, &mut values);

        let out: &T = values.get(&X(0)).unwrap().try_into().ok().unwrap();
        assert_matrix_eq!(
            out.ominus(&p),
            VectorX::zeros(T::DIM),
            comp = abs,
            tol = 1e-6
        );
    }

    pub fn optimize_between<O, T>()
    where
        O: Optimizer,
        T: Variable + Into<VariableEnum>,
        for<'a> &'a VariableEnum: TryInto<&'a T>,
        PriorResidual<T>: Into<ResidualEnum>,
        BetweenResidual<T>: Into<ResidualEnum>,
    {
        let t = VectorX::from_fn(T::DIM, |_, i| ((i as dtype) - (T::DIM as dtype)) / 10.0);
        let p1 = T::exp(t.as_view());

        let t = VectorX::from_fn(T::DIM, |_, i| ((i + 1) as dtype) / 10.0);
        let p2 = T::exp(t.as_view());

        let mut values = Values::new();
        values.insert(X(0), T::identity());
        values.insert(X(1), T::identity());

        let mut graph = Graph::<DefaultBundle>::new();
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
}
