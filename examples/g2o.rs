use plotters::prelude::*;
use samrs::{
    containers::*,
    optimizers::{GaussNewton, Optimizer},
    utils::load_g20,
    variables::*,
};
use std::{env, time::Instant};

// Optimization ideas
// - try_new_from_tripletts - see if there's anyway around a lot of the checks, we should really be fine
// - making duals with exp is SLOW. We should be able to do this faster

fn visualize(init: &Values, sol: &Values) {
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
        .map(|i| (init.get_cast::<SE2>(&X(i)).unwrap()))
        .collect::<Vec<_>>();

    scatter_ctx
        .draw_series(
            init.iter()
                .map(|x| Circle::new((x.x(), x.y()), 2, GREEN.filled())),
        )
        .unwrap();

    let sol = (0..size)
        .map(|i| (sol.get_cast::<SE2>(&X(i)).unwrap()))
        .collect::<Vec<_>>();

    scatter_ctx
        .draw_series(
            sol.iter()
                .map(|x| Circle::new((x.x(), x.y()), 2, BLUE.filled())),
        )
        .unwrap();
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("Usage: {} <g2o file>", args[0]);
        return;
    }
    let filename = &args[1];

    pretty_env_logger::init();

    // Load the graph from the g2o file
    let (graph, init) = load_g20(filename);
    let to_solve = init.clone();
    println!("File loaded");

    // Optimize with GaussNewton
    let mut optimizer: GaussNewton = GaussNewton::new(graph);
    let start = Instant::now();
    let result = optimizer.optimize(to_solve);
    let duration = start.elapsed();

    println!("Optimization took: {:?}", duration);

    match result {
        Ok(_sol) => {
            println!("Optimization successful!");
            // visualize(&init, &sol);
        }
        Err(e) => {
            println!("Optimization failed: {:?}", e);
        }
    }
}
