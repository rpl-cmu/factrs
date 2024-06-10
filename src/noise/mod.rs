use crate::{
    linalg::{MatrixX, VectorX},
    make_enum_noise,
};
pub trait NoiseModel: Sized {
    fn dim(&self) -> usize;

    fn whiten_vec(&self, v: &VectorX) -> VectorX;

    fn whiten_mat(&self, m: &MatrixX) -> MatrixX;
}

mod gaussian;
pub use gaussian::GaussianNoise;

mod macros;

make_enum_noise!(NoiseEnum, GaussianNoise);
