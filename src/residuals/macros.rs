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
            const DIM: usize = 0;

            fn residual_jacobian<K: $crate::containers::Key>(&self, values: &$crate::containers::Values<K, $var>, keys: &[K]) -> $crate::linalg::DiffResult<$crate::linalg::VectorX, $crate::linalg::MatrixX> {
                paste! {
                    match self {
                        $(
                            $name::[<$x $($gen)?>](x) => $crate::residuals::Residual::<$var>::residual_jacobian(x, values, keys),
                        )*
                    }
                }
            }

            fn dim(&self) -> usize {
                paste! {
                    match self {
                        $(
                            $name::[<$x $($gen)?>](x) => $crate::residuals::Residual::<$var>::dim(x),
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
                const DIM: usize = <Self as [<Residual $num>]<V>>::DIM;

                fn residual_jacobian<K: Key>(&self, values: &Values<K, V>, keys: &[K]) -> DiffResult<VectorX, MatrixX> {
                    self.[<residual $num _jacobian>](values, keys)
                }
            }
        }
    };
}
