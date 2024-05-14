#[macro_export]
macro_rules! make_enum_variable {
    ( $name:ident, $( $x:ident),*) => {
        #[derive(Clone, derive_more::Display, derive_more::From, derive_more::TryInto)]
        #[try_into(owned, ref, ref_mut)]
        pub enum $name {
            $(
                $x($x),
            )*
        }

        // Implement the trait for each enum
        impl Variable for $name {
            const DIM: usize = 0;

            fn dim(&self) -> usize {
                match self {
                    $(
                        $name::$x(x) => x.dim(),
                    )*
                }
            }

            fn identity() -> Self {
                panic!("Cannot call static identity on enum")
            }

            fn identity_enum(&self) -> Self {
                match self {
                    $(
                        $name::$x(_) => $name::$x($x::identity()),
                    )*
                }
            }

            fn inverse(&self) -> Self {
                match self {
                    $(
                        $name::$x(x) => $name::$x(x.inverse()),
                    )*
                }
            }

            fn oplus(&self, delta: &VectorD) -> Self {
                match self {
                    $(
                        $name::$x(x) => $name::$x(x.oplus(delta)),
                    )*
                }
            }

            fn ominus(&self, other: &Self) -> VectorD {
                match (self, other) {
                    $(
                        ($name::$x(x1), $name::$x(x2)) => x1.ominus(x2),
                    )*
                    _ => panic!("Cannot subtract different types"),

                }
            }
        }

    };
}
