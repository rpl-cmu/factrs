pub trait Variable: Sized {
    const DIM: usize;
    type TangentVec;

    fn identity() -> Self;

    fn add(&self, delta: &Self::TangentVec) -> Self;

    fn inverse(&self) -> Self;

    fn clone(&self) -> Self;
}

pub trait LieGroup: Variable {
    fn exp(xi: &Self::TangentVec) -> Self;

    fn log(&self) -> Self::TangentVec;

    fn mul(&self, other: Self) -> Self;
}
