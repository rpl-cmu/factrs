#[macro_export]
macro_rules! make_enum_robust {
    // For when we have a single option, and don't need an enum
    ( $name:ident$(,)? ) => {};

    // For implementing the Default trait
    ($name: ident, default= $default:ident, $( $extras:ident),*) => {
        impl Default for $name {
            fn default() -> Self {
                $name::$default($default::default())
            }
        }
    };

    // For making the enum
    ( $name:ident, $( $x:ident),* $(,)?) => {
        #[derive(derive_more::From, derive_more::TryInto)]
        #[try_into(owned, ref, ref_mut)]
        pub enum $name {
            $(
                $x($x),
            )*
        }

        // Implement the trait for each enum
        impl $crate::robust::RobustCost for $name {
            fn loss(&self, d2: $crate::dtype) -> $crate::dtype {
                match self {
                    $(
                        $name::$x(x) => x.loss(d2),
                    )*
                }
            }

            fn weight(&self, d2: $crate::dtype) -> $crate::dtype {
                match self {
                    $(
                        $name::$x(x) => x.weight(d2),
                    )*
                }
            }
        }

        make_enum_robust!($name, default = $( $x),*);
    };
}
