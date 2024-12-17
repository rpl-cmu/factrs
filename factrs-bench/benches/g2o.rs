use diol::prelude::{Bench, BenchConfig, Bencher, black_box, list};
use factrs::{core::GaussNewton, traits::Optimizer, utils::load_g20};

// factrs
fn factrs(bencher: Bencher, file: &str) {
    let (graph, init) = load_g20(file);
    bencher.bench(|| {
        let mut opt: GaussNewton = GaussNewton::new(graph.clone());
        let mut results = opt.optimize(init.clone());
        black_box(&mut results);
    });
}

use std::{collections::HashMap, fs::read_to_string};

use nalgebra as na;
use tiny_solver::{
    factors, gauss_newton_optimizer, loss_functions::HuberLoss,
    optimizer::Optimizer as TSOptimizer, problem,
};

fn read_g2o(filename: &str) -> (problem::Problem, HashMap<String, na::DVector<f64>>) {
    let mut problem = problem::Problem::new();
    let mut init_values = HashMap::<String, na::DVector<f64>>::new();
    for line in read_to_string(filename).unwrap().lines() {
        let line: Vec<&str> = line.split(' ').collect();
        match line[0] {
            "VERTEX_SE2" => {
                let x = line[2].parse::<f64>().unwrap();
                let y = line[3].parse::<f64>().unwrap();
                let theta = line[4].parse::<f64>().unwrap();
                init_values.insert(format!("x{}", line[1]), na::dvector![theta, x, y]);
            }
            "EDGE_SE2" => {
                let id0 = format!("x{}", line[1]);
                let id1 = format!("x{}", line[2]);
                let dx = line[3].parse::<f64>().unwrap();
                let dy = line[4].parse::<f64>().unwrap();
                let dtheta = line[5].parse::<f64>().unwrap();
                // todo add info matrix
                let edge = factors::BetweenFactorSE2 { dx, dy, dtheta };
                problem.add_residual_block(
                    3,
                    &[(&id0, 3), (&id1, 3)],
                    Box::new(edge),
                    Some(Box::new(HuberLoss::new(1.0))),
                );
            }
            _ => {
                println!("err");
                break;
            }
        }
    }
    let origin_factor = factors::PriorFactor {
        v: na::dvector![0.0, 0.0, 0.0],
    };
    problem.add_residual_block(
        3,
        &[("x0", 3)],
        Box::new(origin_factor),
        Some(Box::new(HuberLoss::new(1.0))),
    );
    (problem, init_values)
}

fn tinysolver(bencher: Bencher, file: &str) {
    let (graph, init) = read_g2o(file);
    bencher.bench(|| {
        let gn = gauss_newton_optimizer::GaussNewtonOptimizer::new();
        let mut results = gn.optimize(&graph, &init, None);
        black_box(&mut results);
    });
}

fn main() -> std::io::Result<()> {
    let mut bench = Bench::new(BenchConfig::from_args()?);
    bench.register_many(list![factrs, tinysolver], ["../examples/data/M3500.g2o"]);
    bench.run()?;

    Ok(())
}
