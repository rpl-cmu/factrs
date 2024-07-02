use std::fmt::{Debug, Display};

use crate::{
    containers::{Symbol, Values},
    dtype,
    linalg::{Diff, DiffResult, DimName, MatrixX, Numeric, VectorX},
    variables::{Variable, VariableSafe},
};

type Alias<V, D> = <V as Variable>::Alias<D>;

// ------------------------- Base Residual Trait & Helpers ------------------------- //
pub trait Residual: Debug + Display {
    type DimIn: DimName;
    type DimOut: DimName;
    type NumVars: DimName;

    fn dim_in(&self) -> usize {
        Self::DimIn::USIZE
    }

    fn dim_out(&self) -> usize {
        Self::DimOut::USIZE
    }

    fn residual(&self, values: &Values, keys: &[Symbol]) -> VectorX;

    fn residual_jacobian(&self, values: &Values, keys: &[Symbol]) -> DiffResult<VectorX, MatrixX>;
}

#[cfg_attr(feature = "serde", typetag::serde(tag = "tag"))]
pub trait ResidualSafe: Debug + Display {
    fn dim_in(&self) -> usize;

    fn dim_out(&self) -> usize;

    fn residual(&self, values: &Values, keys: &[Symbol]) -> VectorX;

    fn residual_jacobian(&self, values: &Values, keys: &[Symbol]) -> DiffResult<VectorX, MatrixX>;
}

// ------------------------- Use Macro to create residuals with set sizes -------------------------
use paste::paste;

macro_rules! residual_maker {
    ($num:expr, $( ($idx:expr, $name:ident, $var:ident) ),*) => {
        paste! {
            pub trait [<Residual $num>]: Residual
            {
                $(
                    type $var: Variable<Alias<dtype> = Self::$var> + VariableSafe;
                )*
                type DimIn: DimName;
                type DimOut: DimName;
                type Differ: Diff;

                fn [<residual $num>]<D: Numeric>(&self, $($name: Alias<Self::$var, D>,)*) -> VectorX<D>;

                fn [<residual $num _values>](&self, values: &Values, keys: &[Symbol]) -> VectorX
                where
                    $(
                        Self::$var: 'static,
                    )*
                 {
                    // Unwrap everything
                    $(
                        let $name: &Self::$var = values.get_cast(&keys[$idx]).unwrap_or_else(|| {
                            panic!("Key not found in values: {:?} with type {}", keys[$idx], std::any::type_name::<Self::$var>())
                        });
                    )*
                    self.[<residual $num>]($($name.clone(),)*)
                }


                fn [<residual $num _jacobian>](&self, values: &Values, keys: &[Symbol]) -> DiffResult<VectorX, MatrixX>
                where
                    $(
                        Self::$var: 'static,
                    )*
                {
                    // Unwrap everything
                    $(
                        let $name: &Self::$var = values.get_cast(&keys[$idx]).unwrap_or_else(|| {
                            panic!("Key not found in values: {:?} with type {}", keys[$idx], std::any::type_name::<Self::$var>())
                        });
                    )*
                    Self::Differ::[<jacobian_ $num>](|$($name,)*| self.[<residual $num>]($($name,)*), $($name,)*)
                }
            }
        }
    };
}

residual_maker!(1, (0, v1, V1));
residual_maker!(2, (0, v1, V1), (1, v2, V2));
residual_maker!(3, (0, v1, V1), (1, v2, V2), (2, v3, V3));
residual_maker!(4, (0, v1, V1), (1, v2, V2), (2, v3, V3), (3, v4, V4));
residual_maker!(
    5,
    (0, v1, V1),
    (1, v2, V2),
    (2, v3, V3),
    (3, v4, V4),
    (4, v5, V5)
);
residual_maker!(
    6,
    (0, v1, V1),
    (1, v2, V2),
    (2, v3, V3),
    (3, v4, V4),
    (4, v5, V5),
    (5, v6, V6)
);
