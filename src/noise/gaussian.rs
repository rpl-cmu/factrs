use super::NoiseModel;
use crate::{
    dtype,
    linalg::{Const, Matrix, MatrixView, MatrixViewX, MatrixX, VectorView, VectorViewX, VectorX},
};
use std::fmt;

#[derive(Clone, Debug)]
pub struct GaussianNoise<const N: usize> {
    sqrt_inf: Matrix<N, N>,
}

impl<const N: usize> NoiseModel for GaussianNoise<N> {
    type Dim = Const<N>;

    fn whiten_vec(&self, v: VectorViewX) -> VectorX {
        let mut out = VectorX::zeros(v.len());
        self.sqrt_inf.mul_to(&v, &mut out);
        out
    }

    fn whiten_mat(&self, m: MatrixViewX) -> MatrixX {
        let mut out = MatrixX::zeros(m.nrows(), m.ncols());
        self.sqrt_inf.mul_to(&m, &mut out);
        out
    }
}

impl<const N: usize> GaussianNoise<N> {
    pub fn identity() -> Self {
        let sqrt_inf = Matrix::<N, N>::identity();
        Self { sqrt_inf }
    }

    pub fn from_scalar_sigma(sigma: dtype) -> Self {
        let sqrt_inf = Matrix::<N, N>::from_diagonal_element(1.0 / sigma);
        Self { sqrt_inf }
    }

    pub fn from_scalar_cov(cov: dtype) -> Self {
        let sqrt_inf = Matrix::<N, N>::from_diagonal_element(1.0 / cov.sqrt());
        Self { sqrt_inf }
    }

    pub fn from_diag_sigma(sigma: VectorView<N>) -> Self {
        let sqrt_inf = Matrix::<N, N>::from_diagonal(&sigma.map(|x| 1.0 / x));
        Self { sqrt_inf }
    }

    pub fn from_diag_cov(cov: VectorView<N>) -> Self {
        let sqrt_inf = Matrix::<N, N>::from_diagonal(&cov.map(|x| 1.0 / x.sqrt()));
        Self { sqrt_inf }
    }

    pub fn from_diag_inf(inf: VectorView<N>) -> Self {
        let sqrt_inf = Matrix::<N, N>::from_diagonal(&inf.map(|x| x.sqrt()));
        Self { sqrt_inf }
    }

    pub fn from_matrix_cov(cov: MatrixView<N, N>) -> Self {
        let sqrt_inf = cov
            .try_inverse()
            .expect("Matrix inversion failed when creating sqrt covariance.")
            .cholesky()
            .expect("Cholesky failed when creating sqrt information.")
            .l()
            .transpose();
        Self { sqrt_inf }
    }

    pub fn from_matrix_inf(inf: MatrixView<N, N>) -> Self {
        let sqrt_inf = inf
            .cholesky()
            .expect("Cholesky failed when creating sqrt information.")
            .l()
            .transpose();
        Self { sqrt_inf }
    }
}

impl<const N: usize> fmt::Display for GaussianNoise<N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "GaussianNoise{}: {:}", self.dim(), self.sqrt_inf)
    }
}
