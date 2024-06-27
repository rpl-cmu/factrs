use std::{
    fs::File,
    io::{BufRead, BufReader},
};

use crate::{
    containers::{Graph, Values, X},
    dtype,
    factors::Factor,
    noise::GaussianNoise,
    residuals::{BetweenResidual, PriorResidual},
    variables::*,
};

pub fn load_g20(file: &str) -> (Graph, Values) {
    let file = File::open(file).expect("File not found!");

    let mut values: Values = Values::new();
    let mut graph = Graph::new();

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
                    let factor = Factor::new_base(&[key.clone()], PriorResidual::new(&var.clone()));
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
                let inf = Vector3::new(
                    parts[6].parse::<dtype>().unwrap(),
                    parts[9].parse::<dtype>().unwrap(),
                    parts[11].parse::<dtype>().unwrap(),
                );

                let key1 = X(id_prev);
                let key2 = X(id_curr);
                let var = SE2::new(theta, x, y);
                let noise = GaussianNoise::from_diag_inf(&inf);
                let factor = Factor::new_noise(&[key1, key2], BetweenResidual::new(&var), noise);
                graph.add_factor(factor);
            }
            _ => {
                println!("Unknown line: {}", parts.join(" "));
            }
        }
    }

    (graph, values)
}
