//! Optimizers for solving non-linear least squares problems.
//!
//! Specifically, given a nonlinear least squares problem of the form,
//!
//! $$
//! \Theta^* = \argmin_{\Theta} \sum_{i} \rho_i(||r_i(\Theta)||_{\Sigma_i} )
//! $$
//!
//! These optimizers function by first, linearizing to a linear least squares
//! problem,
//! $$
//! \Delta \Theta = \argmin_{\Delta \Theta} \sum_{i} ||A_i (\Delta \Theta)_i -
//! b_i ||^2
//! $$
//!
//! This can be rearranged to a large, sparse linear system given by,
//!
//! $$
//! A \Delta \Theta = b
//! $$
//!
//! which can then be solved. For example, Gauss-Newton solves this directly,
//! while Levenberg-Marquardt adds a damping term to the diagonal of $A^\top A$
//! to ensure positive definiteness.
//!
//! This module provides a set of optimizers that can be used to solve
//! non-linear least squares problems. Each optimizer implements the [Optimizer]
//! trait to give similar structure and usage.
//!
//! Additionally observers can be added to the optimizer to monitor the progress
//! of the optimization. A prebuilt [Rerun](https://rerun.io/) can be enabled via
//! the `rerun` feature.
//!
//! If you desire to implement your own optimizer, we additionally recommend
//! using the [test_optimizer](crate::test_optimizer) macro to run a handful of
//! simple tests over a few different variable types to ensure correctness.
mod traits;
pub use traits::{
    GraphOptimizer,
    OptError,
    OptObserver,
    OptObserverVec,
    OptParams,
    OptResult,
    Optimizer,
};

mod macros;

mod gauss_newton;
pub use gauss_newton::GaussNewton;

mod levenberg_marquardt;
pub use levenberg_marquardt::LevenMarquardt;

// These aren't tests themselves, but are helpers to test optimizers
#[cfg(test)]
pub mod test {
    use faer::assert_matrix_eq;

    use super::*;
    use crate::{
        containers::{Factor, Graph, Values, X},
        dtype,
        linalg::{Const, VectorX},
        noise::{NoiseModelSafe, UnitNoise},
        residuals::{BetweenResidual, PriorResidual, Residual, ResidualSafe},
        variables::VariableUmbrella,
    };

    pub fn optimize_prior<O, T, const DIM: usize>()
    where
        T: 'static + VariableUmbrella<Dim = Const<DIM>>,
        UnitNoise<DIM>: NoiseModelSafe,
        PriorResidual<T>: ResidualSafe,
        O: Optimizer<Input = Values> + GraphOptimizer,
    {
        let t = VectorX::from_fn(T::DIM, |_, i| ((i + 1) as dtype) / 10.0);
        let p = T::exp(t.as_view());

        let mut values = Values::new();
        values.insert_unchecked(X(0), T::identity());

        let mut graph = Graph::new();
        let res = PriorResidual::new(p.clone());
        let factor = Factor::new_base(&[X(0).into()], res);
        graph.add_factor(factor);

        let mut opt = O::new(graph);
        values = opt.optimize(values).unwrap();

        let out: &T = values.get_unchecked(X(0)).unwrap();
        assert_matrix_eq!(
            out.ominus(&p),
            VectorX::zeros(T::DIM),
            comp = abs,
            tol = 1e-6
        );
    }

    pub fn optimize_between<O, T, const DIM: usize, const DIM_DOUBLE: usize>()
    where
        T: 'static + VariableUmbrella<Dim = nalgebra::Const<DIM>>,
        UnitNoise<DIM>: NoiseModelSafe,
        PriorResidual<T>:
            ResidualSafe + Residual<DimIn = Const<DIM>, DimOut = Const<DIM>, NumVars = Const<1>>,
        BetweenResidual<T>: ResidualSafe
            + Residual<DimIn = Const<DIM_DOUBLE>, DimOut = Const<DIM>, NumVars = Const<2>>,
        O: Optimizer<Input = Values> + GraphOptimizer,
    {
        let t = VectorX::from_fn(T::DIM, |_, i| ((i as dtype) - (T::DIM as dtype)) / 10.0);
        let p1 = T::exp(t.as_view());

        let t = VectorX::from_fn(T::DIM, |_, i| ((i + 1) as dtype) / 10.0);
        let p2 = T::exp(t.as_view());

        let mut values = Values::new();
        values.insert_unchecked(X(0), T::identity());
        values.insert_unchecked(X(1), T::identity());

        let mut graph = Graph::new();
        let res = PriorResidual::new(p1.clone());
        let factor = Factor::new_base(&[X(0).into()], res);
        graph.add_factor(factor);

        let diff = p2.minus(&p1);
        let res = BetweenResidual::new(diff);
        let factor = Factor::new_base(&[X(0).into(), X(1).into()], res);
        graph.add_factor(factor);

        let mut opt = O::new(graph);
        values = opt.optimize(values).unwrap();

        let out1: &T = values.get_unchecked(X(0)).unwrap();
        assert_matrix_eq!(
            out1.ominus(&p1),
            VectorX::zeros(T::DIM),
            comp = abs,
            tol = 1e-6
        );

        let out2: &T = values.get_unchecked(X(1)).unwrap();
        assert_matrix_eq!(
            out2.ominus(&p2),
            VectorX::zeros(T::DIM),
            comp = abs,
            tol = 1e-6
        );
    }
}
