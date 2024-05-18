#[macro_export]
macro_rules! make_enum_variable {
    ( $name:ident, $( $x:ident),*) => {
        #[derive(Clone, Debug, derive_more::Display, derive_more::From, derive_more::TryInto)]
        #[try_into(owned, ref, ref_mut)]
        pub enum $name<D: DualNum = dtype> {
            $(
                $x($x<D>),
            )*
        }

        // Implement the trait for each enum
        impl<D: DualNum> Variable<D> for $name<D> {
            const DIM: usize = 0;
            type Dual = $name<DualVec>;

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

            fn oplus(&self, delta: &VectorD<D>) -> Self {
                match self {
                    $(
                        $name::$x(x) => $name::$x(x.oplus(delta)),
                    )*
                }
            }

            fn ominus(&self, other: &Self) -> VectorD<D> {
                match (self, other) {
                    $(
                        ($name::$x(x1), $name::$x(x2)) => x1.ominus(x2),
                    )*
                    _ => panic!("Cannot subtract different types"),

                }
            }

            fn dual_self(&self) -> Self::Dual {
                match self {
                    $(
                        $name::$x(x) => $name::<DualVec>::$x(x.dual_self()),
                    )*
                }
            }
        }

    };
}
