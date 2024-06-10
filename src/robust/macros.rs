#[macro_export]
macro_rules! make_enum_robust {
    ( $name:ident, $( $x:ident),*) => {
        #[derive(derive_more::From, derive_more::TryInto)]
        #[try_into(owned, ref, ref_mut)]
        pub enum $name {
            $(
                $x($x),
            )*
        }

        // Implement the trait for each enum
        impl $crate::robust::RobustCost for $name {
            fn weight(&self, d2: $crate::dtype) -> $crate::dtype {
                match self {
                    $(
                        $name::$x(x) => x.weight(d2),
                    )*
                }
            }
        }
    };
}
