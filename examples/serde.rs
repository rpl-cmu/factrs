use samrs::{
    containers::{Values, X},
    variables::{SE2, SO2},
};
use serde_json;

fn main() {
    let x = SO2::from_theta(0.6);
    let y = SE2::new(1.0, 2.0, 0.3);
    let mut values = Values::new();
    values.insert(X(0), x);
    values.insert(X(1), y);

    let serialized = serde_json::to_string(&values).unwrap();
    println!("serialized = {}", serialized);

    // Convert the JSON string back to a Point.
    let deserialized: Values = serde_json::from_str(&serialized).unwrap();

    // Prints deserialized = Point { x: 1, y: 2 }
    println!("deserialized = {:?}", deserialized);
}
