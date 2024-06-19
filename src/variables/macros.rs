#[macro_export]
macro_rules! make_enum_variable {
    ( $name:ident$(,)? ) => {};

    ( $name:ident, $( $x:ident),* $(,)?) => {
        #[derive(Clone, derive_more::From, derive_more::TryInto)]
        #[try_into(owned, ref, ref_mut)]
        pub enum $name<D: $crate::linalg::DualNum = $crate::dtype> {
            $(
                $x($x<D>),
            )*
        }

        impl <D: $crate::linalg::DualNum> std::fmt::Display for $name<D> {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                match self {
                    $(
                        $name::$x(x) => write!(f, "{:?}", x),
                    )*
                }
            }
        }

        impl <D: $crate::linalg::DualNum> std::fmt::Debug for $name<D> {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                std::fmt::Display::fmt(self, f)
            }
        }

        // Implement the trait for each enum
        impl<D: $crate::linalg::DualNum> $crate::variables::Variable<D> for $name<D> {
            const DIM: usize = 0;
            type Dual = $name<$crate::linalg::DualVec>;

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

            fn inverse(&self) -> Self {
                match self {
                    $(
                        $name::$x(x) => $name::$x(x.inverse()),
                    )*
                }
            }

            fn compose(&self, other: &Self) -> Self {
                match (self, other) {
                    $(
                        ($name::$x(x), $name::$x(y)) => $name::$x(x.compose(y)),
                    )*
                    _ => panic!("Cannot compose different types"),
                }
            }

            fn exp(_delta: $crate::linalg::VectorViewX<D>) -> Self {
                panic!("Cannot call static exp on enum")
            }

            fn log(&self) -> $crate::linalg::VectorX<D> {
                match self {
                    $(
                        $name::$x(x) => x.log(),
                    )*
                }
            }

            fn dual_self(&self) -> Self::Dual {
                match self {
                    $(
                        $name::$x(x) => $name::<$crate::linalg::DualVec>::$x(x.dual_self()),
                    )*
                }
            }

            // Overrides for various enum helpers
            // For some of these, they work fine without this, but
            // it's preferable we delegate to underlying implementation to reduce # of matches we have to do
            fn identity_enum(&self) -> Self {
                match self {
                    $(
                        $name::$x(_) => $name::$x($x::identity()),
                    )*
                }
            }

            fn oplus(&self, delta: $crate::linalg::VectorViewX<D>) -> Self {
                match self {
                    $(
                        $name::$x(x) => $name::$x(x.oplus(delta)),
                    )*
                }
            }

            fn ominus(&self, other: &Self) -> $crate::linalg::VectorX<D> {
                match (self, other) {
                    $(
                        ($name::$x(x), $name::$x(y)) => x.ominus(y),
                    )*
                    _ => panic!("Cannot ominus different types"),
                }
            }

            fn dual_tangent(&self, idx: usize, total: usize) -> $crate::linalg::VectorX<$crate::linalg::DualVec> {
                match self {
                    $(
                        $name::$x(x) => x.dual_tangent(idx, total),
                    )*
                }
            }

            fn dual(&self, idx: usize, total: usize) -> Self::Dual {
                match self {
                    $(
                        $name::$x(x) => $name::$x(x.dual(idx, total)),
                    )*
                }
            }
        }

    };
}
