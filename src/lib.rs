#[cfg(not(feature = "f32"))]
#[allow(non_camel_case_types)]
pub type dtype = f64;

#[cfg(feature = "f32")]
#[allow(non_camel_case_types)]
pub type dtype = f32;

pub mod containers;
pub mod factors;
pub mod linalg;
pub mod linear;
pub mod noise;
pub mod optimizers;
pub mod residuals;
pub mod robust;
pub mod utils;
pub mod variables;

#[cfg(feature = "rerun")]
pub mod rerun;

#[cfg(feature = "serde")]
pub mod serde {
    pub trait Tagged: serde::Serialize {
        const TAG: &'static str;
    }

    #[macro_export]
    macro_rules! register_typetag {
        ($trait:path, $ty:ty) => {
            // TODO: It'd be great if this was a blanket implementation, but
            // I had problems getting it to run over const generics
            impl $crate::serde::Tagged for $ty {
                const TAG: &'static str = stringify!($ty);
            }

            typetag::__private::inventory::submit! {
                <dyn $trait>::typetag_register(
                    <$ty as $crate::serde::Tagged>::TAG, // Tag of the type
                    (|deserializer| typetag::__private::Result::Ok(
                        typetag::__private::Box::new(
                            typetag::__private::erased_serde::deserialize::<$ty>(deserializer)?
                        ),
                    )) as typetag::__private::DeserializeFn<<dyn $trait as typetag::__private::Strictest>::Object>
                )
            }
        };
    }
}

// Dummy implementation so things don't break when the serde feature is disabled
#[cfg(not(feature = "serde"))]
pub mod serde {
    #[macro_export]
    macro_rules! register_typetag {
        ($trait:path, $ty:ty) => {};
    }
}
