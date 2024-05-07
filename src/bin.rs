use nalgebra::dvector;
use samrs::variables::SO3;
use samrs::variables::{LieGroup, Values, VariableEnum, Vector3, X};

fn main() {
    let xi = dvector![0.1, 0.2, 0.3];

    let r = SO3::exp(&xi);
    let v = Vector3::new(1.0, 2.0, 3.0);
    let e: VariableEnum = r.clone().into();

    let mut values = Values::new();
    values.insert(X(0), r);
    values.insert(X(1), v);
    values.insert(X(2), e);

    println!("{:#?}", values);

    let so3s: Vec<SO3> = values.clone().into();
    println!("{:#?}", so3s);

    let vecs: Vec<Vector3> = values.into();
    println!("{:?}", vecs);
}
