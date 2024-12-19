use diol::prelude::{black_box, list, Bench, BenchConfig, Bencher};
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

fn main() -> std::io::Result<()> {
    let to_run = list![factrs];

    let mut bench = Bench::new(BenchConfig::from_args()?);
    bench.register_many(
        to_run,
        [
            "../examples/data/sphere2500.g2o",
            "../examples/data/parking-garage.g2o",
        ],
    );
    bench.run()?;

    Ok(())
}
