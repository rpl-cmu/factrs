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

// ------------------------- Gtsam ------------------------- //
// https://github.com/sarah-quinones/faer-rs/blob/v0.17.0/faer-bench/src/main.rs
#[cfg(feature = "cpp")]
fn gtsam(bencher: Bencher, file: &str) {
    cxx::let_cxx_string!(cfile = file);
    let gv = factrs_bench::gtsam::load_g2o(&*cfile, true);
    println!("Back in rust land!");
    bencher.bench(|| {
        println!("Running GTSAM");
        factrs_bench::gtsam::run(&gv);
    });
}

fn main() -> std::io::Result<()> {
    #[cfg(feature = "cpp")]
    let to_run = list![gtsam];
    #[cfg(not(feature = "cpp"))]
    let to_run = list![factrs];

    let mut bench = Bench::new(BenchConfig::from_args()?);
    bench.register_many(to_run, [
        "../examples/data/sphere2500.g2o",
        "../examples/data/parking-garage.g2o",
    ]);
    bench.run()?;

    Ok(())
}
