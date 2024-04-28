use nalgebra::{SVector, Scalar, Vector3};

const DIM_MINE: usize = 3;
trait Variable {
    const DIM: usize;

    fn identity() -> Self;

    fn add(&self, other: &SVector<f32, DIM_MINE>) -> Self;

    fn inverse(&self) -> Self;

    fn clone(&self) -> Self;
}

trait LieGroup: Variable {
    fn exp(delta: &SVector<f32, DIM_MINE>) -> Self;

    fn log(&self) -> SVector<f32, DIM_MINE>;

    fn mul(&self, other: Self) -> Self;

    fn add(&self, other: &SVector<f32, DIM_MINE>) -> Self {
        self.mul(Self::exp(other))
    }
}

fn main() {
    // let v: SVector<f32, DIM> = SVector::zeros();
    // println!("{v}");
}
