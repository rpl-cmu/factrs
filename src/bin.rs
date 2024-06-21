use samrs::containers::{GraphGeneric, Symbol, Values, X};
use samrs::factors::FactorGeneric;
use samrs::linalg::dvector;
use samrs::linear::CholeskySolver;
use samrs::make_bundle;
use samrs::noise::GaussianNoise;
use samrs::optimizers::{GaussNewton, Optimizer, OptimizerParams};
use samrs::residuals::PriorResidual;
use samrs::robust::{GemanMcClure, Tukey, L2};
use samrs::variables::{Variable, VariableEnum, SE3, SO3};

fn prior<T: Variable>(p: T) {
    let mut values = Values::new();
    values.insert(X(0), T::identity());

    let mut graph: GraphGeneric<Symbol, T, PriorResidual<T>, GaussianNoise, L2> =
        GraphGeneric::new();
    let res = PriorResidual::new(&p);
    let factor = FactorGeneric::new(vec![X(0)], res).build();
    graph.add_factor(factor);

    let params: OptimizerParams = OptimizerParams::default();
    GaussNewton::optimize(params, &graph, &mut values);

    let out = values.get(&X(0)).unwrap();
}

#[allow(dead_code)]
fn main() {
    let p = SE3::exp(dvector![0.1, 0.2, 0.3, 0.4, 0.5, 0.6].as_view());
    prior(p);
}
