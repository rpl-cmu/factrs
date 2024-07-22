# fact.rs

fact.rs is a nonlinear least squares optimization library over factor graphs, specifically geared for sensor fusion in robotics. It aims to be fast, easy to use, and safe. The fact.rs API takes heavy inspiration from the [gtsam library](https://gtsam.org/).

Currently, it supports the following features
- Gauss-Newton & Levenberg-Marquadt Optimizers
- Common Lie Groups supported (SO2, SO3, SE2, SE3) with optimization in Lie
  Algebras
- Automatic differentiation via dual numbers
- First class support for robust kernels
- Serialization of graphs & variables via optional serde support
- Easy conversion to rerun types for simple visualization

We recommend you checkout the [docs](https://docs.rs/factrs/latest/factrs/) (WIP) for more info.

## Example

```rust
use factrs::prelude::*;

// Make all the values
let mut values = Values::new();
let x = SO2::from_theta(1.0);
let y = SO2::from_theta(2.0);
values.insert(X(0), SO2::identity());
values.insert(X(1), SO2::identity());

// Make the factors & insert into graph
let mut graph = Graph::new();
let res = PriorResidual::new(x.clone());
let factor = Factor::new_base(&[X(0)], res);
graph.add_factor(factor);

let res = BetweenResidual::new(y.minus(&x));
let noise = GaussianNoise::from_scalar_sigma(0.1);
let robust = Huber::default();
let factor = Factor::new_full(&[X(0), X(1)], res, noise, robust);
graph.add_factor(factor);

// Optimize!
let mut opt: GaussNewton = GaussNewton::new(graph);
let result = opt.optimize(values);
```

## TODO
- [ ] IMU Preintegration
- [ ] Sparse solvers via the Bayes Tree
- [ ] iSAM2 implementation
- [ ] Other optimization routines (GNC, Dogleg)
- [ ] Python wrapper
- [ ] Prettify/expand README

## Contributions

Contributions are more than welcome! Feel free to open an issue or a pull request with any ideas, bugs, thoughts, etc you might have. 

We feel rust and robotics are a good match and watch to see rust tooling catch-up to it's C++ counterparts.