use super::NoiseModel;
use crate::dtype;
use crate::linalg::{MatrixX, VectorX};
use std::fmt;

#[derive(Clone, Debug)]
pub struct GaussianNoise {
    sqrt_inf: MatrixX<dtype>,
}

impl NoiseModel for GaussianNoise {
    fn dim(&self) -> usize {
        self.sqrt_inf.shape().0
    }

    fn whiten_vec(&self, v: &VectorX) -> VectorX {
        &self.sqrt_inf * v
    }

    fn whiten_mat(&self, m: &MatrixX) -> MatrixX {
        &self.sqrt_inf * m
    }
}

impl GaussianNoise {
    pub fn identity(n: usize) -> Self {
        let sqrt_inf = MatrixX::<dtype>::identity(n, n);
        Self { sqrt_inf }
    }

    pub fn from_scalar_sigma(sigma: dtype, n: usize) -> Self {
        let sqrt_inf = MatrixX::<dtype>::from_diagonal_element(n, n, 1.0 / sigma);
        Self { sqrt_inf }
    }

    pub fn from_scalar_cov(cov: dtype, n: usize) -> Self {
        let sqrt_inf = MatrixX::<dtype>::from_diagonal_element(n, n, 1.0 / cov.sqrt());
        Self { sqrt_inf }
    }

    pub fn from_diag_sigma(sigma: &VectorX) -> Self {
        let sqrt_inf = MatrixX::<dtype>::from_diagonal(&sigma.map(|x| 1.0 / x));
        Self { sqrt_inf }
    }

    pub fn from_diag_cov(cov: &VectorX) -> Self {
        let sqrt_inf = MatrixX::<dtype>::from_diagonal(&cov.map(|x| 1.0 / x.sqrt()));
        Self { sqrt_inf }
    }

    pub fn from_matrix_cov(cov: &MatrixX<dtype>) -> Self {
        // TODO: Double check if I want upper or lower triangular cholesky
        let sqrt_inf = cov
            .clone()
            .try_inverse()
            .expect("Matrix inversion failed when creating sqrt covariance.")
            .cholesky()
            .expect("Cholesky failed when creating sqrt covariance.")
            .l();
        Self { sqrt_inf }
    }
}

impl fmt::Display for GaussianNoise {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "GaussianNoise{}: {}", self.dim(), self.sqrt_inf)
    }
}
