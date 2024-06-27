use crate::containers::{Symbol, Values};
use crate::linalg::{Diff, DiffResult, Dim, DualVec, MatrixX, VectorX};
use crate::variables::Variable;
use paste::paste;
use std::fmt;

type DualVar<V> = <V as Variable>::Dual;

// ------------------------- Base Residual Trait & Helpers ------------------------- //
pub trait Residual: fmt::Debug {
    type DimOut: Dim;
    type NumVars: Dim;

    fn dim_out(&self) -> usize {
        Self::DimOut::try_to_usize().unwrap()
    }

    fn residual(&self, values: &Values, keys: &[Symbol]) -> VectorX;

    fn residual_jacobian(&self, values: &Values, keys: &[Symbol]) -> DiffResult<VectorX, MatrixX>;
}

pub trait ResidualSafe {
    fn residual(&self, values: &Values, keys: &[Symbol]) -> VectorX;

    fn residual_jacobian(&self, values: &Values, keys: &[Symbol]) -> DiffResult<VectorX, MatrixX>;
}

impl<T: Residual> ResidualSafe for T {
    fn residual(&self, values: &Values, keys: &[Symbol]) -> VectorX {
        self.residual(values, keys)
    }

    fn residual_jacobian(&self, values: &Values, keys: &[Symbol]) -> DiffResult<VectorX, MatrixX> {
        self.residual_jacobian(values, keys)
    }
}

// ------------------------- Use Macro to create residuals with set sizes ------------------------- //
macro_rules! residual_maker {
    ($num:expr, $( ($idx:expr, $name:ident, $var:ident) ),*) => {
        paste! {
            pub trait [<Residual $num>]: Residual
            {
                $(
                    type $var: Variable;
                )*
                type DimOut: Dim;
                type Differ: Diff;

                fn [<residual $num>](&self, $($name: DualVar<Self::$var>,)*) -> VectorX<DualVec>;

                fn [<residual $num _single>](&self, values: &Values, keys: &[Symbol]) -> VectorX
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
                    self.[<residual $num>]($($name.dual_self(),)*).map(|r| r.re)
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
