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
        impl $crate::traits::Residual<$var> for $name {
            const DIM: usize = 0;

            fn residual(&self, v: &[<$var as $crate::traits::Variable>::Dual]) -> $crate::linalg::VectorX<$crate::traits::DualVec> {
                paste! {
                    match self {
                        $(
                            $name::[<$x $($gen)?>](x) => $crate::traits::Residual::<$var>::residual(x, v),
                        )*
                    }
                }
            }
        }

    };
}
