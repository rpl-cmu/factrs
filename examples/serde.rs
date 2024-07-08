use factrs::{
    containers::{Graph, Values, X},
    factors::Factor,
    noise::GaussianNoise,
    residuals::{BetweenResidual, PriorResidual},
    robust::{GemanMcClure, Huber, L2},
    variables::{SE2, SO2},
};

fn main() {
    // ------------------------- Try with values ------------------------- //
    let x = SO2::from_theta(0.6);
    let y = SE2::new(1.0, 2.0, 0.3);
    let mut values = Values::new();
    values.insert(X(0), x.clone());
    values.insert(X(1), y.clone());

    let serialized = serde_json::to_string_pretty(&values).unwrap();
    println!("serialized = {}", serialized);

    // Convert the JSON string back to a Point.
    let deserialized: Values = serde_json::from_str(&serialized).unwrap();
    println!("deserialized = {}", deserialized);

    // ------------------------- Try with graph ------------------------- //
    let prior = PriorResidual::new(x);
    let bet = BetweenResidual::new(y);

    let prior = Factor::new_full(
        &[X(0)],
        prior,
        GaussianNoise::from_scalar_cov(0.1),
        GemanMcClure::default(),
    );
    let bet = Factor::new_full(
        &[X(0), X(1)],
        bet,
        GaussianNoise::from_scalar_cov(10.0),
        Huber::default(),
    );
    let mut graph = Graph::new();
    graph.add_factor(prior);
    graph.add_factor(bet);

    let serialized = serde_json::to_string_pretty(&graph).unwrap();
    println!("serialized = {}", serialized);

    let deserialized: Graph = serde_json::from_str(&serialized).unwrap();
    println!("deserialized = {:?}", graph);
}
