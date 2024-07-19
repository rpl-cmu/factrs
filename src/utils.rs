use std::{
    fs::File,
    io::{BufRead, BufReader},
};

use crate::{
    containers::{Factor, Graph, Values, X},
    dtype,
    linalg::{Matrix3, Matrix6, Vector3},
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
                    let factor = Factor::new_base(&[key.clone()], PriorResidual::new(var.clone()));
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

                let m11 = parts[6].parse::<dtype>().unwrap();
                let m12 = parts[7].parse::<dtype>().unwrap();
                let m13 = parts[8].parse::<dtype>().unwrap();
                let m22 = parts[9].parse::<dtype>().unwrap();
                let m23 = parts[10].parse::<dtype>().unwrap();
                let m33 = parts[11].parse::<dtype>().unwrap();
                let inf = Matrix3::new(m11, m12, m13, m12, m22, m23, m13, m23, m33);

                let key1 = X(id_prev);
                let key2 = X(id_curr);
                let var = SE2::new(theta, x, y);
                let noise = GaussianNoise::from_matrix_inf(inf.as_view());
                let factor = Factor::new_noise(&[key1, key2], BetweenResidual::new(var), noise);
                graph.add_factor(factor);
            }

            "VERTEX_SE3:QUAT" => {
                let id = parts[1].parse::<u64>().unwrap();
                let x = parts[2].parse::<dtype>().unwrap();
                let y = parts[3].parse::<dtype>().unwrap();
                let z = parts[4].parse::<dtype>().unwrap();
                let qx = parts[5].parse::<dtype>().unwrap();
                let qy = parts[6].parse::<dtype>().unwrap();
                let qz = parts[7].parse::<dtype>().unwrap();
                let qw = parts[8].parse::<dtype>().unwrap();

                let rot = SO3::from_xyzw(qx, qy, qz, qw);
                let xyz = Vector3::new(x, y, z);
                let var = SE3::from_rot_trans(rot, xyz);
                let key = X(id);

                // Add prior on whatever the first variable is
                if values.len() == 1 {
                    let noise =
                        GaussianNoise::<6>::from_diag_covs(1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4);
                    let factor =
                        Factor::new_noise(&[key.clone()], PriorResidual::new(var.clone()), noise);
                    graph.add_factor(factor);
                }

                values.insert(key, var);
            }

            "EDGE_SE3:QUAT" => {
                let id_prev = parts[1].parse::<u64>().unwrap();
                let id_curr = parts[2].parse::<u64>().unwrap();
                let x = parts[3].parse::<dtype>().unwrap();
                let y = parts[4].parse::<dtype>().unwrap();
                let z = parts[5].parse::<dtype>().unwrap();
                let qx = parts[6].parse::<dtype>().unwrap();
                let qy = parts[7].parse::<dtype>().unwrap();
                let qz = parts[8].parse::<dtype>().unwrap();
                let qw = parts[9].parse::<dtype>().unwrap();

                let m11 = parts[10].parse::<dtype>().unwrap();
                let m12 = parts[11].parse::<dtype>().unwrap();
                let m13 = parts[12].parse::<dtype>().unwrap();
                let m14 = parts[13].parse::<dtype>().unwrap();
                let m15 = parts[14].parse::<dtype>().unwrap();
                let m16 = parts[15].parse::<dtype>().unwrap();
                let m22 = parts[16].parse::<dtype>().unwrap();
                let m23 = parts[17].parse::<dtype>().unwrap();
                let m24 = parts[18].parse::<dtype>().unwrap();
                let m25 = parts[19].parse::<dtype>().unwrap();
                let m26 = parts[20].parse::<dtype>().unwrap();
                let m33 = parts[21].parse::<dtype>().unwrap();
                let m34 = parts[22].parse::<dtype>().unwrap();
                let m35 = parts[23].parse::<dtype>().unwrap();
                let m36 = parts[24].parse::<dtype>().unwrap();
                let m44 = parts[25].parse::<dtype>().unwrap();
                let m45 = parts[26].parse::<dtype>().unwrap();
                let m46 = parts[27].parse::<dtype>().unwrap();
                let m55 = parts[28].parse::<dtype>().unwrap();
                let m56 = parts[29].parse::<dtype>().unwrap();
                let m66 = parts[30].parse::<dtype>().unwrap();
                #[rustfmt::skip]
                let inf = Matrix6::new(
                    m44, m45, m46, m14, m15, m16,
                    m45, m55, m56, m24, m25, m26,
                    m46, m56, m66, m34, m35, m36,
                    m14, m24, m34, m11, m12, m13,
                    m15, m25, m35, m12, m22, m23,
                    m16, m26, m36, m13, m23, m33,
                );

                let rot = SO3::from_xyzw(qx, qy, qz, qw);
                let xyz = Vector3::new(x, y, z);
                let var = SE3::from_rot_trans(rot, xyz);

                // println!("var: {:?}", var);
                // println!("2499 {:?}", values.get_cast::<SE3>(&X(2499)).unwrap());
                // panic!();

                let key1 = X(id_prev);
                let key2 = X(id_curr);
                let noise = GaussianNoise::from_matrix_inf(inf.as_view());
                let factor = Factor::new_noise(&[key1, key2], BetweenResidual::new(var), noise);
                graph.add_factor(factor);
            }

            _ => {
                println!(",Unknown line: {}", parts.join(" "));
            }
        }
    }

    (graph, values)
}
