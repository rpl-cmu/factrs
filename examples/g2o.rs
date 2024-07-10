use std::{env, time::Instant};

use factrs::{
    optimizers::{GaussNewton, GraphOptimizer, Optimizer},
    utils::load_g20,
};
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
        }
        Err(e) => {
            println!("Optimization failed: {:?}", e);
        }
    }
}
