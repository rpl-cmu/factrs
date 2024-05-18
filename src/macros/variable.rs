#[macro_export]
macro_rules! make_enum_variable {
    ( $name:ident, $( $x:ident),*) => {
        #[derive(Clone, derive_more::From, derive_more::TryInto)]
        #[try_into(owned, ref, ref_mut)]
        pub enum $name<D: $crate::traits::DualNum = $crate::dtype> {
            $(
                $x($x<D>),
            )*
        }

        impl <D: $crate::traits::DualNum> std::fmt::Display for $name<D> {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                match self {
                    $(
                        $name::$x(x) => write!(f, "{:?}", x),
                    )*
                }
            }
        }

        impl <D: $crate::traits::DualNum> std::fmt::Debug for $name<D> {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                std::fmt::Display::fmt(self, f)
            }
        }

        // Implement the trait for each enum
        impl<D: $crate::traits::DualNum> $crate::traits::Variable<D> for $name<D> {
            const DIM: usize = 0;
            type Dual = $name<$crate::traits::DualVec>;

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

            fn oplus(&self, delta: &$crate::linalg::VectorX<D>) -> Self {
                match self {
                    $(
                        $name::$x(x) => $name::$x(x.oplus(delta)),
                    )*
                }
            }

            fn ominus(&self, other: &Self) -> $crate::linalg::VectorX<D> {
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
                        $name::$x(x) => $name::<$crate::traits::DualVec>::$x(x.dual_self()),
                    )*
                }
            }
        }

    };
}
