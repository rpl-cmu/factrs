use plotters::prelude::*;
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

// Optimization ideas
// - try_new_from_tripletts - see if there's anyway around a lot of the checks, we should really be fine
// - making duals with exp is SLOW. We should be able to do this faster
// TODO: Name bundle versions GraphBundled, ValuesBundled, etc, and the templated versions Graph, Values, etc

fn visualize(init: &Values<Symbol, SE2>, sol: &Values<Symbol, SE2>) {
    let root_drawing_area = BitMapBackend::new("m3500_rs.png", (1024, 1024)).into_drawing_area();
    root_drawing_area.fill(&WHITE).unwrap();

    let mut scatter_ctx = ChartBuilder::on(&root_drawing_area)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(-50f64..50f64, -80f64..20f64)
        .unwrap();
    scatter_ctx
        .configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .draw()
        .unwrap();

    // Draw the initial points
    let size = init.len() as u64;
    let init = (0..size)
        .map(|i| (init.get(&X(i)).unwrap()))
        .collect::<Vec<_>>();

    scatter_ctx
        .draw_series(
            init.iter()
                .map(|x| Circle::new((x.x(), x.y()), 2, GREEN.filled())),
        )
        .unwrap();

    let sol = (0..size)
        .map(|i| (sol.get(&X(i)).unwrap()))
        .collect::<Vec<_>>();

    scatter_ctx
        .draw_series(
            sol.iter()
                .map(|x| Circle::new((x.x(), x.y()), 2, BLUE.filled())),
        )
        .unwrap();
}

fn main() {
    pretty_env_logger::init();

    // Load the graph from the g2o file
    let (graph, init) = load_g20::<M3500Bundle>("./examples/M3500.g2o");
    let to_solve = init.clone();
    println!("File loaded");

    // Optimize with GaussNewton
    let mut optimizer: GaussNewtonBundled<M3500Bundle> = GaussNewton::new(graph);
    let start = Instant::now();
    let result = optimizer.optimize(to_solve);
    let duration = start.elapsed();

    println!("Optimization took: {:?}", duration);

    match result {
        Ok(sol) => {
            println!("Optimization successful!");
            visualize(&init, &sol);
        }
        Err(e) => {
            println!("Optimization failed: {:?}", e);
        }
    }
}
