#[macro_export]
macro_rules! make_enum_residual {
    ( $name:ident, $( $x:ident $(< $gen:ident >)? ),*) => {
        #[derive(Clone, Debug, derive_more::Display, derive_more::From, derive_more::TryInto)]
        #[try_into(owned, ref, ref_mut)]
        pub enum $name {
            $(
                $x($x$(<$gen>)?),
            )*
        }

        // Implement the trait for each enum
        impl<V: $crate::traits::Variable<$crate::dtype>> $crate::traits::Residual<V> for $name {
            const DIM: usize = 0;

            fn residual(&self, v: &[V::Dual]) -> $crate::linalg::VectorX<$crate::traits::DualVec> {
                match self {
                    $(
                        $name::$x(x) => x.residual(v),
                    )*
                }
            }
        }

    };
}
