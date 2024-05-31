#[macro_export]
macro_rules! make_enum_noise {
    ( $name:ident, $( $x:ident),*) => {
        #[derive(Clone, Debug, derive_more::Display, derive_more::From, derive_more::TryInto)]
        #[try_into(owned, ref, ref_mut)]
        pub enum $name {
            $(
                $x($x),
            )*
        }

        // Implement the trait for each enum
        impl $crate::traits::NoiseModel for $name {
            const DIM: usize = 0;

            fn dim(&self) -> usize {
                match self {
                    $(
                        $name::$x(x) => x.dim(),
                    )*
                }
            }

            fn whiten(&self, v: &$crate::linalg::VectorX) -> $crate::linalg::VectorX {
                match self {
                    $(
                        $name::$x(x) => x.whiten(v),
                    )*
                }
            }
        }

    };
}
