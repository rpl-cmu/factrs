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

# Example

```rust
use factrs::{
   assign_symbols,
   fac,
   containers::{FactorBuilder, Graph, Values},
   noise::GaussianNoise,
   optimizers::GaussNewton,
   residuals::{BetweenResidual, PriorResidual},
   robust::Huber,
   traits::*,
   variables::SO2,
};

// Assign symbols to variable types
assign_symbols!(X: SO2);

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
let noise = GaussianNoise::from_scalar_sigma(0.1);
let robust = Huber::default();
let factor = fac![res, (X(0), X(1)), noise, robust];
// The same as above, but verbose
// let factor = FactorBuilder::new2(res, X(0), X(1))
//     .noise(noise)
//     .robust(robust)
//     .build();
graph.add_factor(factor);

// Optimize!
let mut opt: GaussNewton = GaussNewton::new(graph);
let result = opt.optimize(values);
```

# Installation
Simply add via cargo as you do any rust dependency,
```bash
cargo add factrs
```

# Contributions

Contributions are more than welcome! Feel free to open an issue or a pull request with any ideas, bugs, features, etc you might have or want. 

We feel rust and robotics are a good match and want to see rust robotics libraries catch-up to their C++ counterparts.