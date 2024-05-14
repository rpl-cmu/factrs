use nalgebra::dvector;
use samrs::traits::{LieGroup, Variable};
use samrs::variables::SO3;
use samrs::variables::{Symbol, Values, VariableEnum, Vector3, X};

fn main() {
    let xi = dvector![0.1, 0.2, 0.3];

    let r = SO3::exp(&xi);
    let v = Vector3::new(1.0, 2.0, 3.0);
    let e: VariableEnum = r.clone().into();

    let mut values: Values<Symbol, VariableEnum> = Values::new();
    values.insert(X(0), r);
    values.insert(X(1), v);
    values.insert(X(2), e);
    println!("{:#?}", values);

    let temp = values.get_mut(&X(0)).unwrap();
    *temp = temp.oplus(&xi);

    let so3s: Vec<&SO3> = values.filter().collect();
    println!("{:#?}", so3s);

    let vecs: Vec<Vector3> = values.into_filter().collect();
    println!("{:#?}", vecs);
}
