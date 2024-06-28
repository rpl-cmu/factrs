use super::{Dyn, RealField};
use crate::dtype;

// ------------------------- Enum for all Dtypes ------------------------- //
#[derive(derive_more::From)]
pub enum DTypes {
    Float(dtype),
    DualVectorX(DualVectorX),
}

impl From<DTypes> for f64 {
    fn from(d: DTypes) -> Self {
        match d {
            DTypes::Float(f) => f,
            DTypes::DualVectorX(d) => d.re,
        }
    }
}

impl From<DTypes> for DualVectorX {
    fn from(d: DTypes) -> Self {
        match d {
            DTypes::Float(f) => DualVectorX::from_re(f),
            DTypes::DualVectorX(d) => d,
        }
    }
}

// Setup dual num
pub trait Numeric: RealField + num_dual::DualNum<dtype> + From<DTypes> + Into<DTypes> {}
impl<G: RealField + num_dual::DualNum<dtype> + From<DTypes> + Into<DTypes>> Numeric for G {}

pub type DualVectorX = num_dual::DualVec<dtype, dtype, Dyn>;
pub type DualVector<N> = num_dual::DualVec<dtype, dtype, N>;
pub type DualScalar = num_dual::Dual<dtype, dtype>;
