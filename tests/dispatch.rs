use samrs::traits::Variable;
use samrs::variables::{Vector3, SO3};
use samrs::{self, make_enum_variable};

make_enum_variable!(MyVariables, SO3, Vector3);

#[test]
fn dispatchable() {
    let unique: SO3 = Variable::identity();

    println!("unique: {:?}", unique);
    let _test = MyVariables::SO3(unique.clone());

    let _unique: MyVariables = MyVariables::from(unique);
}