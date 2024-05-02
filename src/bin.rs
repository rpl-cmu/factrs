use nalgebra::Vector3;
use samrs::variables::SO3;
use samrs::variables::{LieGroup, Variable, X};

fn main() {
    let xi = Vector3::new(0.1, 0.2, 0.3);

    let r = SO3::exp(&xi);
    let xi_out = r.log();

    println!("xi: {:?}", xi);
    println!("xi_out: {:?}", xi_out);

    let double = r.oplus(&xi);
    let xi_double = double.log();
    println!("xi_double: {:?}", xi_double);

    let x = X(5);
    println!("x: {}", x);
}
