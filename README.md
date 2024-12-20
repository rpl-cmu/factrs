# factrs

[![crates.io](https://img.shields.io/crates/v/factrs.svg)](https://crates.io/crates/factrs)
[![ci](https://github.com/rpl-cmu/factrs/actions/workflows/rust.yml/badge.svg)](https://github.com/rpl-cmu/factrs/actions/workflows/rust.yml)
[![docs.rs](https://docs.rs/factrs/badge.svg)](https://docs.rs/factrs)

factrs is a nonlinear least squares optimization library over factor graphs written in Rust.

It is specifically geared toward sensor fusion in robotics. It aims to be fast, easy to use, and safe. The factrs API takes heavy inspiration from the [gtsam library](https://gtsam.org/).

Currently, it supports the following features
- Gauss-Newton & Levenberg-Marquadt Optimizers
- Common Lie Groups supported (SO2, SO3, SE2, SE3) with optimization in Lie
  Algebras
- Pose graph optimization and IMU preintegration
- Automatic differentiation via dual numbers
- First class support for robust kernels
- Serialization of graphs & variables via optional serde support
- Easy conversion to rerun types for straightforward visualization

We recommend you checkout the [docs](https://docs.rs/factrs/latest/factrs/) for more info.

# Examples
There's a number of examples found in the [examples](/examples/) folder, including loading g20 files, serialization, and custom factors.

To run any of the examples, simply clone this repository and run,
```bash
cargo run --release --example g20 ./examples/data/M3500.g20
```
to visualize the optimization steps with [rerun](https://rerun.io) simply add `--features rerun` to the above command.

Running the other examples can be done similarly,
```bash
cargo run --release --example serde --features serde
cargo run --release --example gps
``` 

Additionally, we recommend checking out the [tests](/tests/) folder for more examples of custom noise models, residuals, robust kernels, and variables.

<details>
<summary>Full Example</summary>

```rust
use factrs::{
    core::{
        assign_symbols, fac, BetweenResidual, GaussNewton, Graph, Huber, PriorResidual, Values, SO2,
    },
    traits::*,
};

// Assign symbols to variable types
assign_symbols!(X: SO2);

fn main() {
    // Make all the values
    let mut values = Values::new();

    let x = SO2::from_theta(1.0);
    let y = SO2::from_theta(2.0);
    values.insert(X(0), SO2::identity());
    values.insert(X(1), SO2::identity());

    // Make the factors & insert into graph
    let mut graph = Graph::new();
    let res = PriorResidual::new(x.clone());
    let factor = fac![res, X(0)];
    graph.add_factor(factor);

    let res = BetweenResidual::new(y.minus(&x));
    let robust = Huber::default();
    let factor = fac![res, (X(0), X(1)), 0.1 as std, robust];
    // fac! is syntactic sugar for the following
    // let noise = GaussianNoise::from_scalar_sigma(0.1);
    // let factor = FactorBuilder::new2(res, X(0), X(1))
    //     .noise(noise)
    //     .robust(robust)
    //     .build();
    graph.add_factor(factor);

    // Optimize!
    let mut opt: GaussNewton = GaussNewton::new(graph);
    let result = opt.optimize(values).unwrap();
    println!("Results {:#}", result);
}
```

# Installation
Simply add via cargo as you do any rust dependency,
```bash
cargo add factrs
```
</details>


# Contributions

Contributions are more than welcome! Feel free to open an issue or a pull request with any ideas, bugs, features, etc you might have or want. 

We feel rust and robotics are a good match and want to see rust robotics libraries catch-up to their C++ counterparts.