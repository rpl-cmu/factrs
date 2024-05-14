use samrs::traits::Variable;
use samrs::variables::{Vector3, VectorD, SO3};
use samrs::{self, make_enum_variable};

make_enum_variable!(MyVariables, SO3, Vector3);

#[test]
fn dispatchable() {
    let unique: SO3 = Variable::identity();
    let vec3: Vector3 = Variable::identity();

    println!("unique: {:?}", unique);
    let test = MyVariables::SO3(unique.clone());

    let unique: MyVariables = MyVariables::from(unique);
}
