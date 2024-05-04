use nalgebra::dvector;
use samrs::variables::SO3;
use samrs::variables::{LieGroup, Variable, VariableEnum, Vector3, X};

fn main() {
    let xi = dvector![0.1, 0.2, 0.3];

    let r = SO3::exp(&xi);
    let xi_out = r.log();

    println!("xi: {:?}", xi);
    println!("xi_out: {:?}", xi_out);

    let double = r.oplus(&xi);
    let xi_double = double.log();
    println!("xi_double: {:?}", xi_double);

    let x = X(5);
    println!("x: {}", x);
    println!("");

    let en: VariableEnum = r.into();
    let ragain: SO3 = en.try_into().unwrap();
    println!("ragain: {:?}", ragain);
}
