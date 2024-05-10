use crate::variables::{
    Variable, Vector1, Vector10, Vector2, Vector3, Vector4, Vector5, Vector6, Vector7, Vector8,
    Vector9, VectorD, SE3, SO3,
};
use derive_more::Display;
use enum_dispatch::enum_dispatch;

// This is a slightly modified version of Variable, just a little less convenient
// However it is in the form we need for enum_dispatch
#[enum_dispatch]
pub trait DispatchableVariable {
    // const DIM: usize;

    fn dim(&self) -> usize;

    fn identity(&self) -> Self;

    fn inverse(&self) -> Self;

    fn oplus(&self, delta: &VectorD) -> Self;

    // Ominus may be doable, but harder due to enum in call type
    // fn ominus(&self, other: &Enum) -> VectorD;
}

// Default implementation for all of our variables
impl<T: Variable> DispatchableVariable for T {
    fn identity(&self) -> Self {
        T::identity()
    }

    fn dim(&self) -> usize {
        T::DIM
    }

    fn inverse(&self) -> Self {
        Variable::inverse(self)
    }

    fn oplus(&self, delta: &VectorD) -> Self {
        Variable::oplus(self, delta)
    }
}

#[enum_dispatch(DispatchableVariable)]
#[derive(Clone, Display)]
pub enum VariableEnum {
    SO3,
    SE3,
    Vector1,
    Vector2,
    Vector3,
    Vector4,
    Vector5,
    Vector6,
    Vector7,
    Vector8,
    Vector9,
    Vector10,
}
