mod traits;
pub use traits::{OptError, OptResult, Optimizer, OptimizerParams};

mod macros;

mod gauss_newton;
pub use gauss_newton::GaussNewton;

mod levenberg_marquardt;
pub use levenberg_marquardt::LevenMarquardt;

// These aren't tests themselves, but are helpers to test optimizers
#[cfg(test)]
pub mod test {
    use faer::assert_matrix_eq;
    use nalgebra::{allocator::Allocator, DefaultAllocator, DimNameAdd, DimNameSum};

    use super::*;
    use crate::{
        containers::{Graph, Values, X},
        dtype,
        factors::Factor,
        linalg::{AllocatorBuffer, Const, DualAllocator, DualVector, VectorX},
        residuals::{BetweenResidual, PriorResidual},
        variables::Variable,
    };

    pub fn optimize_prior<O, T, const DIM: usize>()
    where
        T: 'static + Variable<Dim = nalgebra::Const<DIM>, Alias<dtype> = T>,
        O: Optimizer,
    {
        let t = VectorX::from_fn(T::DIM, |_, i| ((i + 1) as dtype) / 10.0);
        let p = T::exp(t.as_view());

        let mut values = Values::new();
        values.insert(X(0), T::identity());

        let mut graph = Graph::new();
        let res = PriorResidual::new(p.clone());
        let factor = Factor::new_base(&[X(0)], res);
        graph.add_factor(factor);

        let mut opt = O::new(graph);
        values = opt.optimize(values).unwrap();

        let out: &T = values.get_cast(&X(0)).unwrap();
        assert_matrix_eq!(
            out.ominus(&p),
            VectorX::zeros(T::DIM),
            comp = abs,
            tol = 1e-6
        );
    }

    // pub fn optimize_between<O, T, const DIM: usize>()
    // where
    //     T: 'static + Variable<Dim = nalgebra::Const<DIM>, Alias<dtype> = T>,
    //     O: Optimizer,
    //     Const<DIM>: DimNameAdd<Const<DIM>>,
    //     DualVector<DimNameSum<Const<DIM>, Const<DIM>>>: Copy,
    // {
    //     let t = VectorX::from_fn(T::DIM, |_, i| ((i as dtype) - (T::DIM as dtype)) / 10.0);
    //     let p1 = T::exp(t.as_view());

    //     let t = VectorX::from_fn(T::DIM, |_, i| ((i + 1) as dtype) / 10.0);
    //     let p2 = T::exp(t.as_view());

    //     let mut values = Values::new();
    //     values.insert(X(0), T::identity());
    //     values.insert(X(1), T::identity());

    //     let mut graph = Graph::new();
    //     let res = PriorResidual::new(p1.clone());
    //     let factor = Factor::new_base(&[X(0)], res);
    //     graph.add_factor(factor);

    //     let diff = p2.minus(&p1);
    //     let res = BetweenResidual::new(diff);
    //     let factor = Factor::new_base(&[X(0), X(1)], res);
    //     graph.add_factor(factor);

    //     let mut opt = O::new(graph);
    //     values = opt.optimize(values).unwrap();

    //     let out1: &T = values.get_cast(&X(0)).unwrap();
    //     assert_matrix_eq!(
    //         out1.ominus(&p1),
    //         VectorX::zeros(T::DIM),
    //         comp = abs,
    //         tol = 1e-6
    //     );

    //     let out2: &T = values.get_cast(&X(1)).unwrap();
    //     assert_matrix_eq!(
    //         out2.ominus(&p2),
    //         VectorX::zeros(T::DIM),
    //         comp = abs,
    //         tol = 1e-6
    //     );
    // }
}
