use crate::{
    containers::{Symbol, Values},
    dtype,
    linalg::{
        Diff, DiffResult, DimName, DualAllocator, DualVector, DualVectorGeneric, DualVectorX,
        MatrixX, Numeric, VectorX,
    },
    variables::Variable,
};
use nalgebra::{
    allocator::Allocator,
    constraint::{DimEq, ShapeConstraint},
    DefaultAllocator,
};
use paste::paste;
use std::fmt;

type Alias<V, D> = <V as Variable>::Alias<D>;

// ------------------------- Base Residual Trait & Helpers ------------------------- //
pub trait Residual: fmt::Debug {
    type DimIn: DimName;
    type DimOut: DimName;
    type NumVars: DimName;

    fn dim_out(&self) -> usize {
        Self::DimOut::USIZE
    }

    fn residual(&self, values: &Values, keys: &[Symbol]) -> VectorX;

    fn residual_jacobian(&self, values: &Values, keys: &[Symbol]) -> DiffResult<VectorX, MatrixX>
    where
        <DefaultAllocator as Allocator<dtype, Self::DimIn>>::Buffer: Sync + Send,
        DefaultAllocator: DualAllocator<Self::DimIn>;
}

pub trait ResidualSafe {
    fn dim_in(&self) -> usize;

    fn dim_out(&self) -> usize;

    fn residual(&self, values: &Values, keys: &[Symbol]) -> VectorX;

    fn residual_jacobian(&self, values: &Values, keys: &[Symbol]) -> DiffResult<VectorX, MatrixX>;
}

impl<T: Residual> ResidualSafe for T
where
    <DefaultAllocator as Allocator<dtype, T::DimIn>>::Buffer: Sync + Send,
    DefaultAllocator: DualAllocator<T::DimIn>,
{
    fn dim_in(&self) -> usize {
        T::DimIn::USIZE
    }

    fn dim_out(&self) -> usize {
        T::DimOut::USIZE
    }

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
                    type $var: Variable<Alias<dtype> = Self::$var>;
                )*
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

pub trait Residual1: Residual {
    type Differ: Diff;
    type V1: Variable<Alias<dtype> = Self::V1>;
    type DimOut: DimName;
    type DimIn: DimName;

    fn residual1<D: Numeric>(&self, v1: Alias<Self::V1, D>) -> VectorX<D>;

    fn residual1_values(&self, values: &Values, keys: &[Symbol]) -> VectorX
    where
        Self::V1: 'static,
    {
        let v1: &Self::V1 = values.get_cast(&keys[0]).unwrap_or_else(|| {
            panic!(
                "Key not found in values: {:?} with type {}",
                keys[0],
                std::any::type_name::<Self::V1>()
            )
        });
        self.residual1(v1.clone())
    }
    fn residual1_jacobian(&self, values: &Values, keys: &[Symbol]) -> DiffResult<VectorX, MatrixX>
    where
        Self::V1: 'static,
        <DefaultAllocator as Allocator<dtype, <Self as Residual1>::DimIn>>::Buffer: Sync + Send,
        DefaultAllocator: DualAllocator<<Self as Residual1>::DimIn>,
        DualVectorGeneric<<Self as Residual1>::DimIn>: Copy,
    {
        let v1: &Self::V1 = values.get_cast(&keys[0]).unwrap_or_else(|| {
            panic!(
                "Key not found in values: {:?} with type {}",
                keys[0],
                std::any::type_name::<Self::V1>()
            )
        });
        Self::Differ::jacobian_1(|v1| self.residual1(v1), v1)
    }
}
pub trait Residual2: Residual {
    type Differ: Diff;
    type V1: Variable<Alias<dtype> = Self::V1>;
    type V2: Variable<Alias<dtype> = Self::V2>;
    type DimIn: DimName;
    type DimOut: DimName;
    fn residual2<D: Numeric>(&self, v1: Alias<Self::V1, D>, v2: Alias<Self::V2, D>) -> VectorX<D>;

    fn residual2_values(&self, values: &Values, keys: &[Symbol]) -> VectorX
    where
        Self::V1: 'static,
        Self::V2: 'static,
    {
        let v1: &Self::V1 = values.get_cast(&keys[0]).unwrap_or_else(|| {
            panic!(
                "Key not found in values: {:?} with type {}",
                keys[0],
                std::any::type_name::<Self::V1>()
            );
        });
        let v2: &Self::V2 = values.get_cast(&keys[1]).unwrap_or_else(|| {
            panic!(
                "Key not found in values: {:?} with type {}",
                keys[1],
                std::any::type_name::<Self::V2>()
            );
        });
        self.residual2(v1.clone(), v2.clone())
    }
    fn residual2_jacobian(&self, values: &Values, keys: &[Symbol]) -> DiffResult<VectorX, MatrixX>
    where
        Self::V1: 'static,
        Self::V2: 'static,
        <DefaultAllocator as Allocator<dtype, <Self as Residual2>::DimIn>>::Buffer: Sync + Send,
        DefaultAllocator: DualAllocator<<Self as Residual2>::DimIn>,
        DualVectorGeneric<<Self as Residual2>::DimIn>: Copy,
    {
        let v1: &Self::V1 = values.get_cast(&keys[0]).unwrap_or_else(|| {
            panic!(
                "Key not found in values: {:?} with type {}",
                keys[0],
                std::any::type_name::<Self::V1>()
            );
        });
        let v2: &Self::V2 = values.get_cast(&keys[1]).unwrap_or_else(|| {
            panic!(
                "Key not found in values: {:?} with type {}",
                keys[1],
                std::any::type_name::<Self::V2>()
            );
        });

        Self::Differ::jacobian_2(|v1, v2| self.residual2(v1, v2), v1, v2)
    }
}

// residual_maker!(3, (0, v1, V1), (1, v2, V2), (2, v3, V3));
// residual_maker!(4, (0, v1, V1), (1, v2, V2), (2, v3, V3), (3, v4, V4));
// residual_maker!(
//     5,
//     (0, v1, V1),
//     (1, v2, V2),
//     (2, v3, V3),
//     (3, v4, V4),
//     (4, v5, V5)
// );
// residual_maker!(
//     6,
//     (0, v1, V1),
//     (1, v2, V2),
//     (2, v3, V3),
//     (3, v4, V4),
//     (4, v5, V5),
//     (5, v6, V6)
// );
