#[macro_export]
macro_rules! make_enum_residual {
    ( $name:ident, $var:ident, $(,)? ) => {};

    ( $name:ident, $var:ident, $( $x:ident $(< $gen:ident >)? ),* $(,)?) => {
        use paste::paste;

        paste! {
            #[derive(Clone, Debug, derive_more::Display, derive_more::From, derive_more::TryInto)]
            #[try_into(owned, ref, ref_mut)]
            pub enum $name {
                $(
                    [<$x $($gen)?>]($x$(<$gen>)?),
                )*
            }
        }

        // Implement the trait for the specified enum
        impl $crate::residuals::Residual<$var> for $name {
            type NumVars = $crate::linalg::Const<0>;
            type DimOut = $crate::linalg::Const<0>;

            fn residual<K: $crate::containers::Key>(&self, values: &$crate::containers::Values<K, $var>, keys: &[K]) -> $crate::linalg::VectorX {
                paste! {
                    match self {
                        $(
                            $name::[<$x $($gen)?>](x) => $crate::residuals::Residual::<$var>::residual(x, values, keys),
                        )*
                    }
                }
            }

            fn residual_jacobian<K: $crate::containers::Key>(&self, values: &$crate::containers::Values<K, $var>, keys: &[K]) -> $crate::linalg::DiffResult<$crate::linalg::VectorX, $crate::linalg::MatrixX> {
                paste! {
                    match self {
                        $(
                            $name::[<$x $($gen)?>](x) => $crate::residuals::Residual::<$var>::residual_jacobian(x, values, keys),
                        )*
                    }
                }
            }

            fn dim_out(&self) -> usize {
                paste! {
                    match self {
                        $(
                            $name::[<$x $($gen)?>](x) => $crate::residuals::Residual::<$var>::dim_out(x),
                        )*
                    }
                }
            }
        }

    };
}

#[macro_export]
macro_rules! impl_residual {
    ($num:expr, $name:ident $(< $T:ident : $Trait:ident >)?, $($vars:ident),* ) => {
        use paste::paste;
        paste!{
            impl<V: Variable, $($T: $Trait)?> Residual<V> for $name $(< $T >)?
            where
                $(
                    for<'a> &'a V: std::convert::TryInto<&'a $vars>,
                )*
            {
                type DimOut = <Self as [<Residual $num>]<V>>::DimOut;
                type NumVars = $crate::linalg::Const<$num>;

                fn residual<K: Key>(&self, values: &Values<K, V>, keys: &[K]) -> VectorX {
                    self.[<residual $num _single>](values, keys)
                }

                fn residual_jacobian<K: Key>(&self, values: &Values<K, V>, keys: &[K]) -> DiffResult<VectorX, MatrixX> {
                    self.[<residual $num _jacobian>](values, keys)
                }
            }
        }
    };
}
