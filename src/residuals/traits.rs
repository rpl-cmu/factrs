use std::fmt::{Debug, Display};

use crate::{
    containers::{Symbol, Values},
    linalg::{Diff, DiffResult, DimName, MatrixX, Numeric, VectorX},
    variables::{Variable, VariableUmbrella},
};

type Alias<V, D> = <V as Variable>::Alias<D>;

// ------------------ Base Residual Trait & Helpers ------------------ //
/// Base trait for residuals
///
/// This trait is used to implement custom residuals. It is recommended to use
/// one of the numbered residuals traits instead, and then call the
/// [impl_residual](crate::impl_residual) macro to implement this trait.
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

/// The object safe version of [Residual].
///
/// This trait is used to allow for dynamic dispatch of residuals.
/// Implemented for all types that implement [Residual].
#[cfg_attr(feature = "serde", typetag::serde(tag = "tag"))]
pub trait ResidualSafe: Debug + Display {
    fn dim_in(&self) -> usize;

    fn dim_out(&self) -> usize;

    fn residual(&self, values: &Values, keys: &[Symbol]) -> VectorX;

    fn residual_jacobian(&self, values: &Values, keys: &[Symbol]) -> DiffResult<VectorX, MatrixX>;
}

impl<
        #[cfg(not(feature = "serde"))] T: Residual,
        #[cfg(feature = "serde")] T: Residual + crate::serde::Tagged,
    > ResidualSafe for T
{
    fn dim_in(&self) -> usize {
        Residual::dim_in(self)
    }

    fn dim_out(&self) -> usize {
        Residual::dim_out(self)
    }

    fn residual(&self, values: &Values, keys: &[Symbol]) -> VectorX {
        Residual::residual(self, values, keys)
    }

    fn residual_jacobian(&self, values: &Values, keys: &[Symbol]) -> DiffResult<VectorX, MatrixX> {
        Residual::residual_jacobian(self, values, keys)
    }

    #[doc(hidden)]
    #[cfg(feature = "serde")]
    fn typetag_name(&self) -> &'static str {
        Self::TAG
    }

    #[doc(hidden)]
    #[cfg(feature = "serde")]
    fn typetag_deserialize(&self) {}
}

// -------------- Use Macro to create residuals with set sizes -------------- //
use paste::paste;

macro_rules! residual_maker {
    ($num:expr, $( ($idx:expr, $name:ident, $var:ident) ),*) => {
        paste! {
            #[doc=concat!("Residual trait for ", $num, " variables")]
            pub trait [<Residual $num>]: Residual
            {
                $(
                    #[doc=concat!("Type of variable ", $idx)]
                    type $var: VariableUmbrella;
                )*
                /// The total input dimension
                type DimIn: DimName;
                /// The output dimension of the residual
                type DimOut: DimName;
                /// Differentiator type (see [Diff](crate::linalg::Diff))
                type Differ: Diff;

                /// Main residual computation
                ///
                /// If implementing your own residual, this is the only method you need to implement.
                /// It is generic over the dtype to allow for differentiable types.
                fn [<residual $num>]<D: Numeric>(&self, $($name: Alias<Self::$var, D>,)*) -> VectorX<D>;

                #[doc=concat!("Wrapper that unpacks variables and calls [", stringify!([<residual $num>]), "](Self::", stringify!([<residual $num>]), ").")]
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


                #[doc=concat!("Wrapper that unpacks variables and computes jacobians using [", stringify!([<residual $num>]), "](Self::", stringify!([<residual $num>]), ").")]
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
