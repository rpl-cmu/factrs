mod traits;
pub use traits::{OptError, OptResult, Optimizer, OptimizerParams};

mod macros;

mod gauss_newton;
pub use gauss_newton::{GaussNewton, GaussNewtonBundled};

mod levenberg_marquardt;
pub use levenberg_marquardt::LevenMarquardt;

// These aren't tests themselves, but are helpers to test optimizers
#[cfg(test)]
pub mod test {
    use faer::assert_matrix_eq;

    use crate::{
        bundle::DefaultBundle,
        containers::{Graph, Symbol, Values, X},
        dtype,
        factors::FactorGeneric,
        linalg::VectorX,
        noise::NoiseEnum,
        residuals::{BetweenResidual, PriorResidual, ResidualEnum},
        robust::RobustEnum,
        variables::{Variable, VariableEnum},
    };

    use super::*;

    pub fn optimize_prior<O, T>()
    where
        T: Variable + Into<VariableEnum>,
        for<'a> &'a VariableEnum: TryInto<&'a T>,
        PriorResidual<T>: Into<ResidualEnum>,
        O: Optimizer<Symbol, VariableEnum, ResidualEnum, NoiseEnum, RobustEnum>,
    {
        let t = VectorX::from_fn(T::DIM, |_, i| ((i + 1) as dtype) / 10.0);
        let p = T::exp(t.as_view());

        let mut values = Values::new();
        values.insert(X(0), T::identity());

        let mut graph = Graph::<DefaultBundle>::new();
        let res = PriorResidual::new(&p);
        let factor = FactorGeneric::new(vec![X(0)], res).build();
        graph.add_factor(factor);

        let mut opt = O::new(graph);
        values = opt.optimize(values).unwrap();

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
        O: Optimizer<Symbol, VariableEnum, ResidualEnum, NoiseEnum, RobustEnum>,
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

        let mut opt = O::new(graph);
        values = opt.optimize(values).unwrap();

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
