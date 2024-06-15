use crate::{
    containers::{Key, Symbol},
    dtype,
    noise::{NoiseEnum, NoiseModel},
    residuals::{Residual, ResidualEnum},
    robust::{RobustCost, RobustEnum},
    variables::{Variable, VariableEnum},
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
    type Key = Symbol;
    type Variable = VariableEnum;
    type Robust = RobustEnum;
    type Noise = NoiseEnum;
    type Residual = ResidualEnum;
}
