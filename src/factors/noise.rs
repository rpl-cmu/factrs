use crate::traits::NoiseModel;
use crate::variables::VectorD;
use nalgebra::DMatrix;
use std::fmt;

#[derive(Clone, Debug)]
pub struct GaussianNoise<const N: usize> {
    pub sqrt_cov: DMatrix<f64>,
}

impl<const N: usize> NoiseModel for GaussianNoise<N> {
    const DIM: usize = N;

    fn whiten(&self, v: &VectorD) -> VectorD {
        &self.sqrt_cov * v
    }
}

impl<const N: usize> GaussianNoise<N> {
    pub fn from_scalar_sigma(sigma: f64) -> Self {
        let sqrt_cov = DMatrix::<f64>::from_diagonal_element(N, N, sigma);
        Self { sqrt_cov }
    }

    pub fn from_scalar_cov(cov: f64) -> Self {
        let sqrt_cov = DMatrix::<f64>::from_diagonal_element(N, N, cov.sqrt());
        Self { sqrt_cov }
    }

    pub fn from_diag_sigma(sigma: &VectorD) -> Self {
        let sqrt_cov = DMatrix::<f64>::from_diagonal(sigma);
        Self { sqrt_cov }
    }

    pub fn from_diag_cov(cov: &VectorD) -> Self {
        let sqrt_cov = DMatrix::<f64>::from_diagonal(&cov.map(|x| x.sqrt()));
        Self { sqrt_cov }
    }

    pub fn from_matrix_sigma(sigma: &DMatrix<f64>) -> Self {
        let sqrt_cov = sigma.clone();
        Self { sqrt_cov }
    }

    pub fn from_matrix_cov(cov: &DMatrix<f64>) -> Self {
        // TODO: Double check if I want upper or lower triangular cholesky
        let sqrt_cov = cov
            .clone()
            .cholesky()
            .expect("Cholesky failed when creating sqrt covariance.")
            .l();
        Self { sqrt_cov }
    }
}

impl<const N: usize> fmt::Display for GaussianNoise<N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "GaussianNoise1: {}", self.sqrt_cov)
    }
}

pub type GaussianNoise1 = GaussianNoise<1>;
pub type GaussianNoise2 = GaussianNoise<2>;
pub type GaussianNoise3 = GaussianNoise<3>;
pub type GaussianNoise4 = GaussianNoise<4>;
pub type GaussianNoise5 = GaussianNoise<5>;
pub type GaussianNoise6 = GaussianNoise<6>;
pub type GaussianNoise7 = GaussianNoise<7>;
pub type GaussianNoise8 = GaussianNoise<8>;
pub type GaussianNoise9 = GaussianNoise<9>;
pub type GaussianNoise10 = GaussianNoise<10>;
