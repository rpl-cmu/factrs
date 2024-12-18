use cxx::let_cxx_string;

fn main() {
    println!("Hello, world!");
    factrs_bench::gtsam::hello();
    let_cxx_string!(file = "data/2d/1000.g2o");
    let gv = factrs_bench::gtsam::load_g2o(&file, false);
}
