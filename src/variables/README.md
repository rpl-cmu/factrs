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