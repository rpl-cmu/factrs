use crate::{
    containers::Graph,
    dtype,
    factors::Factor,
    noise::{NoiseEnum, NoiseModel},
    residuals::{Residual, ResidualEnum},
    robust::{RobustCost, RobustEnum},
    traits::{Key, Variable},
    variables,
};

// Trait
pub trait Bundle: Sized {
    type Key: Key;
    type Variable: Variable<dtype>;
    type Robust: RobustCost;
    type Noise: NoiseModel;
    type Residual: Residual<Self::Variable>;
}

// Default Implementation that uses everything we have
pub struct DefaultBundle;

impl Bundle for DefaultBundle {
    type Key = variables::Symbol;
    type Variable = variables::VariableEnum;
    type Robust = RobustEnum;
    type Noise = NoiseEnum;
    type Residual = ResidualEnum;
}

// Some type aliases to make life easier
pub type FactorBundled<B = DefaultBundle> = Factor<
    <B as Bundle>::Key,
    <B as Bundle>::Variable,
    <B as Bundle>::Residual,
    <B as Bundle>::Noise,
    <B as Bundle>::Robust,
>;

pub type GraphBundled<B> = Graph<
    <B as Bundle>::Key,
    <B as Bundle>::Variable,
    <B as Bundle>::Residual,
    <B as Bundle>::Noise,
    <B as Bundle>::Robust,
>;
