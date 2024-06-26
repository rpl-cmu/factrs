use samrs::containers::*;
use samrs::make_bundle;
use samrs::noise::*;
use samrs::optimizers::GaussNewton;
use samrs::optimizers::GaussNewtonBundled;
use samrs::optimizers::Optimizer;
use samrs::residuals::*;
use samrs::robust::*;
use samrs::utils::load_g20;
use samrs::variables::*;
use std::time::Instant;

make_bundle!(
    M3500Bundle;
    Symbol;
    SE2;
    L2;
    GaussianNoise;
    MRes: PriorResidual<SE2>, BetweenResidual<SE2>;
);

// TODO: Visualize this tomorrow!
fn main() {
    // TODO: Name bundle versions GraphBundled, ValuesBundled, etc, and the templated versions Graph, Values, etc
    let (graph, values) = load_g20::<M3500Bundle>("./examples/M3500.g2o");
    println!("File loaded");

    // Optimize with GaussNewton
    // TODO: Add GaussNewtonBundled type
    let mut optimizer: GaussNewtonBundled<M3500Bundle> = GaussNewton::new(graph);
    let start = Instant::now();
    let result = optimizer.optimize(values);
    let duration = start.elapsed();

    println!("Optimization took: {:?}", duration);

    match result {
        Ok(values) => {
            println!("Optimization successful!");
            println!("X0: {}", values.get(&X(0)).unwrap())
            // for (key, value) in values.iter() {
            //     println!("{}: {}", key, value);
            // }
        }
        Err(e) => {
            println!("Optimization failed: {:?}", e);
        }
    }
}
