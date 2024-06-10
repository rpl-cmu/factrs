use crate::{dtype, linalg::VectorX, make_enum_robust};

pub trait RobustCost: Sized {
    fn weight(&self, d2: dtype) -> dtype;

    fn weight_vector(&self, r: &VectorX) -> VectorX {
        r * self.weight(r.norm_squared()).sqrt()
    }
}

mod robust;
pub use robust::*;

mod macros;

make_enum_robust!(
    RobustEnum,
    L2,
    L1,
    Huber,
    Fair,
    Cauchy,
    GemanMcClure,
    Welsch,
    Tukey
);
