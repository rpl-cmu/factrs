use std::fs::File;
use std::io::{BufRead, BufReader};

use nalgebra::dvector;

use crate::bundle::Bundle;
use crate::containers::{Graph, Symbol, Values, X};
use crate::factors::Factor;
use crate::noise::GaussianNoise;
use crate::residuals::{BetweenResidual, PriorResidual};
use crate::robust::L2;
use crate::{dtype, variables::*};

pub fn load_g20<B: Bundle<Key = Symbol>>(file: &str) -> (Graph<B>, Values<B::Key, B::Variable>)
where
    B::Variable: From<SE2>,
    B::Residual: From<BetweenResidual<SE2>>,
    B::Residual: From<PriorResidual<SE2>>,
    B::Noise: From<GaussianNoise>,
    B::Robust: From<L2>,
{
    let file = File::open(file).expect("File not found!");

    let mut values: Values<Symbol, B::Variable> = Values::new();
    let mut graph = Graph::<B>::new();

    for line in BufReader::new(file).lines() {
        let line = line.unwrap();
        let parts = line.split(" ").collect::<Vec<&str>>();
        match parts[0] {
            "VERTEX_SE2" => {
                let id = parts[1].parse::<u64>().unwrap();
                let x = parts[2].parse::<dtype>().unwrap();
                let y = parts[3].parse::<dtype>().unwrap();
                let theta = parts[4].parse::<dtype>().unwrap();

                let var = SE2::new(theta, x, y);
                let key = X(id);

                // Add prior on whatever the first variable is
                if values.len() == 1 {
                    let factor =
                        Factor::<B>::new(vec![key.clone()], PriorResidual::new(&var.clone()))
                            .build();
                    graph.add_factor(factor);
                }

                values.insert(key, var);
            }

            "EDGE_SE2" => {
                let id_prev = parts[1].parse::<u64>().unwrap();
                let id_curr = parts[2].parse::<u64>().unwrap();
                let x = parts[3].parse::<dtype>().unwrap();
                let y = parts[4].parse::<dtype>().unwrap();
                let theta = parts[5].parse::<dtype>().unwrap();
                // TODO: Handle non diagonal
                let inf = dvector![
                    parts[6].parse::<dtype>().unwrap(),
                    parts[9].parse::<dtype>().unwrap(),
                    parts[11].parse::<dtype>().unwrap(),
                ];

                let key1 = X(id_prev);
                let key2 = X(id_curr);
                let var = SE2::new(theta, x, y);
                let noise = GaussianNoise::from_diag_inf(&inf);
                let factor = Factor::<B>::new(vec![key1, key2], BetweenResidual::new(&var))
                    .set_noise(noise)
                    .build();
                graph.add_factor(factor);
            }
            _ => {
                println!("Unknown line: {}", parts.join(" "));
            }
        }
    }

    (graph, values)
}
