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

#[macro_export]
macro_rules! make_bundle {
    ($bundle_name:ident;
        $key:ident;
        $var_name:ident$(:)? $($var_args:ident),*;
        $robust_name:ident$(:)? $($robust_args:ident),*;
        $noise_name:ident$(:)? $($noise_args:ident),*;
        $residual_name:ident$(< $gen:ident >)?$(:)? $( $residual_args:ident $(< $residual_types:ident >)? ),*$(;)?) => {

        pub struct $bundle_name;

        $crate::make_enum_variable!($var_name, $($var_args),*);
        $crate::make_enum_robust!($robust_name, $($robust_args),*);
        $crate::make_enum_noise!($noise_name, $($noise_args),*);
        // This technically drops the generic from residual name in the case that we're not making an enum
        // But it's a dry case, so it probably fine
        $crate::make_enum_residual!($residual_name, $var_name, $( $residual_args $(< $residual_types >)? ),*);

        impl Bundle for $bundle_name {
            type Key = $key;
            type Variable = $var_name;
            type Robust = $robust_name;
            type Noise = $noise_name;
            type Residual = $residual_name$(< $gen >)?;
        }
    };
}
