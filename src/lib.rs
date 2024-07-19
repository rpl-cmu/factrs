//! fact.rs is a nonlinear least squares optimization library over factor
//! graphs, specifically geared for sensor fusion in robotics.
//!
//! Currently, it supports the following features
//! - Gauss-Newton & Levenberg-Marquadt Optimizers
//! - Common Lie Groups supported (SO2, SO3, SE2, SE3) with optimization in Lie
//!   Algebras
//! - Automatic differentiation via dual numbers
//! - First class support for robust kernels
//! - Serialization of graphs & variables via optional serde support
//! - Easy conversion to rerun types for simple visualization
//!
//! # Example
//! ```
//! use factrs::prelude::*;
//!
//! // Make all the values
//! let mut values = Values::new();
//!
//! let x = SO2::from_theta(1.0);
//! let y = SO2::from_theta(2.0);
//! values.insert(X(0), SO2::identity());
//! values.insert(X(1), SO2::identity());
//!
//! // Make the factors & insert into graph
//! let mut graph = Graph::new();
//!
//! let res = PriorResidual::new(x.clone());
//! let factor = Factor::new_base(&[X(0)], res);
//! graph.add_factor(factor);
//!
//! let res = BetweenResidual::new(y.minus(&x));
//! let noise = GaussianNoise::from_scalar_sigma(0.1);
//! let robust = Huber::default();
//! let factor = Factor::new_full(&[X(0), X(1)], res, noise, robust);
//! graph.add_factor(factor);
//!
//! // Optimize!
//! let mut opt: GaussNewton = GaussNewton::new(graph);
//! let result = opt.optimize(values);
//! ```

#[cfg(not(feature = "f32"))]
#[allow(non_camel_case_types)]
pub type dtype = f64;

#[cfg(feature = "f32")]
#[allow(non_camel_case_types)]
pub type dtype = f32;

pub mod containers;
pub mod linalg;
pub mod linear;
pub mod noise;
pub mod optimizers;
pub mod residuals;
pub mod robust;
pub mod utils;
pub mod variables;

pub mod prelude {
    pub use crate::{
        containers::*,
        noise::*,
        optimizers::*,
        residuals::*,
        robust::*,
        variables::*,
    };
}

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
