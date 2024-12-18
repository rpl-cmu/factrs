use diol::prelude::{Bench, BenchConfig, Bencher, black_box, list};
// ------------------------- factrs ------------------------- //
use factrs::{core::GaussNewton, traits::Optimizer, utils::load_g20};
fn factrs(bencher: Bencher, file: &str) {
    let (graph, init) = load_g20(file);
    bencher.bench(|| {
        let mut opt: GaussNewton = GaussNewton::new(graph.clone());
        let mut results = opt.optimize(init.clone());
        black_box(&mut results);
    });
}

// ------------------------- tiny-solver ------------------------- //
use tiny_solver::{gauss_newton_optimizer, optimizer::Optimizer as TSOptimizer};

fn tinysolver(bencher: Bencher, file: &str) {
    let (graph, init) = factrs_bench::tiny_solver::load_g2o(file);
    bencher.bench(|| {
        let gn = gauss_newton_optimizer::GaussNewtonOptimizer::new();
        let mut results = gn.optimize(&graph, &init, None);
        black_box(&mut results);
    });
}

// ------------------------- Gtsam ------------------------- //
// https://github.com/sarah-quinones/faer-rs/blob/v0.17.0/faer-bench/src/main.rs
#[cfg(feature = "cpp")]
fn gtsam(bencher: Bencher, file: &str) {
    cxx::let_cxx_string!(cfile = file);
    let gv = factrs_bench::gtsam::load_g2o(&*cfile, false);
    bencher.bench(|| {
        factrs_bench::gtsam::run(&gv);
    });
}

fn main() -> std::io::Result<()> {
    #[cfg(feature = "cpp")]
    let to_run = list![factrs, tinysolver, gtsam];
    #[cfg(not(feature = "cpp"))]
    let to_run = list![factrs, tinysolver];

    let mut bench = Bench::new(BenchConfig::from_args()?);
    bench.register_many(to_run, ["../examples/data/M3500.g2o"]);
    bench.run()?;

    Ok(())
}
