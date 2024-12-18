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

// ------------------------- Tiny Solver ------------------------- //
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
// TODO: Figure out how to load / pass the file to the C++ function
// TODO: We have to be careful as we don't want to accidentally measure the time
// it takes to load the file
// #[cfg(feature = "cpp")]
// mod gtsam {
//     unsafe extern "C" {
//         fn gtsam(file: *const std::os::raw::c_char);
//     }
// }

// ------------------------- Ceres ------------------------- //
// #[cfg(feature = "cpp")]
// mod ceres {
//     unsafe extern "C" {
//         fn ceres(file: *const std::os::raw::c_char);
//     }
// }

fn main() -> std::io::Result<()> {
    let mut bench = Bench::new(BenchConfig::from_args()?);
    bench.register_many(list![factrs, tinysolver], ["../examples/data/M3500.g2o"]);
    bench.run()?;

    Ok(())
}
