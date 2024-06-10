#[macro_export]
macro_rules! make_enum_residual {
    ( $name:ident, $var:ident, $( $x:ident $(< $gen:ident >)? ),*) => {
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

            fn residual_jacobian<K: $crate::traits::Key>(&self, values: &$crate::containers::Values<K, $var>, keys: &[K]) -> ($crate::linalg::VectorX, $crate::linalg::MatrixX) {
                paste! {
                    match self {
                        $(
                            $name::[<$x $($gen)?>](x) => $crate::residuals::Residual::<$var>::residual_jacobian(x, values, keys),
                        )*
                    }
                }
            }
        }

    };
}
