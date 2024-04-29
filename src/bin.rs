mod variables;

use nalgebra::Vector3;
use variables::SO3;
use variables::{LieGroup, Variable};

fn main() {
    let xi = Vector3::new(0.1, 0.2, 0.3);

    let r = SO3::exp(&xi);
    let xi_out = r.log();

    println!("xi: {:?}", xi);
    println!("xi_out: {:?}", xi_out);

    let double = r.add(&xi);
    let xi_double = double.log();
    println!("xi_double: {:?}", xi_double);
}
