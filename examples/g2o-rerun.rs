use std::{
    env,
    net::{SocketAddr, SocketAddrV4},
    time::Instant,
};

use factrs::{
    optimizers::{GaussNewton, GraphOptimizer, Optimizer},
    rerun::RerunObserver,
    utils::load_g20,
    variables::*,
};
use rerun::{Arrows2D, Arrows3D, Points2D, Points3D};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ---------------------- Parse Arguments & Load data ---------------------- //
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("Usage: {} <g2o file> <points/arrows>", args[0]);
        return Ok(());
    }

    let filename = &args[1];
    let obj = if args.len() >= 3 { &args[2] } else { "arrows" };

    pretty_env_logger::init();

    // Load the graph from the g2o file
    let (graph, init) = load_g20(filename);
    println!("File loaded");
    let dim = if init.filter::<SE2>().count() != 0 {
        "se2"
    } else if init.filter::<SE3>().count() != 0 {
        "se3"
    } else {
        panic!("Graph doesn't have SE2 or SE3 variables");
    };

    // ------------------------- Connect to rerun ------------------------- //
    // Setup the rerun & the callback
    let socket = SocketAddrV4::new("172.31.65.81".parse()?, 9876);
    let rec = rerun::RecordingStreamBuilder::new("rerun_example_dna_abacus")
        .connect_opts(SocketAddr::V4(socket), rerun::default_flush_timeout())?;

    let mut optimizer: GaussNewton = GaussNewton::new(graph);
    let topic = "base/solution";
    match (dim, obj) {
        ("se2", "points") => {
            let callback = RerunObserver::<SE2, Points2D>::new(rec, topic);
            optimizer.observers.add(callback)
        }
        ("se2", "arrows") => {
            let callback = RerunObserver::<SE2, Arrows2D>::new(rec, topic);
            optimizer.observers.add(callback)
        }
        ("se3", "points") => {
            let callback = RerunObserver::<SE3, Points3D>::new(rec, topic);
            optimizer.observers.add(callback)
        }
        ("se3", "arrows") => {
            let callback = RerunObserver::<SE3, Arrows3D>::new(rec, topic);
            optimizer.observers.add(callback)
        }
        _ => panic!("Invalid arguments"),
    };

    // ------------------------- Optimize ------------------------- //
    let start = Instant::now();
    let _result = optimizer.optimize(init);
    let duration = start.elapsed();

    println!("Optimization took: {:?}", duration);
    Ok(())
}
