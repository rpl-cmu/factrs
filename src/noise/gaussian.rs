use std::fmt;

use super::{NoiseModel, UnitNoise};
use crate::{
    dtype,
    linalg::{Const, Matrix, MatrixView, MatrixX, Vector, VectorView, VectorX},
    register_noise,
};

register_noise!(
    GaussianNoise<1>,
    GaussianNoise<2>,
    GaussianNoise<3>,
    GaussianNoise<4>,
    GaussianNoise<5>,
    GaussianNoise<6>,
    GaussianNoise<7>,
    GaussianNoise<8>,
    GaussianNoise<9>,
    GaussianNoise<10>,
    GaussianNoise<11>,
    GaussianNoise<12>,
);

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GaussianNoise<const N: usize> {
    sqrt_inf: Matrix<N, N>,
}

impl<const N: usize> NoiseModel for GaussianNoise<N> {
    type Dim = Const<N>;

    fn whiten_vec(&self, v: VectorX) -> VectorX {
        let mut out = VectorX::zeros(v.len());
        self.sqrt_inf.mul_to(&v, &mut out);
        out
    }

    fn whiten_mat(&self, m: MatrixX) -> MatrixX {
        let mut out = MatrixX::zeros(m.nrows(), m.ncols());
        self.sqrt_inf.mul_to(&m, &mut out);
        out
    }
}

impl<const N: usize> GaussianNoise<N> {
    pub fn identity() -> UnitNoise<N> {
        UnitNoise
    }

    pub fn from_scalar_sigma(sigma: dtype) -> Self {
        let sqrt_inf = Matrix::<N, N>::from_diagonal_element(1.0 / sigma);
        Self { sqrt_inf }
    }

    pub fn from_scalar_cov(cov: dtype) -> Self {
        let sqrt_inf = Matrix::<N, N>::from_diagonal_element(1.0 / cov.sqrt());
        Self { sqrt_inf }
    }

    pub fn from_vec_sigma(sigma: VectorView<N>) -> Self {
        let sqrt_inf = Matrix::<N, N>::from_diagonal(&sigma.map(|x| 1.0 / x));
        Self { sqrt_inf }
    }

    pub fn from_vec_cov(cov: VectorView<N>) -> Self {
        let sqrt_inf = Matrix::<N, N>::from_diagonal(&cov.map(|x| 1.0 / x.sqrt()));
        Self { sqrt_inf }
    }

    pub fn from_vec_inf(inf: VectorView<N>) -> Self {
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

macro_rules! make_gaussian_vector {
    ($num:expr, [$($args:ident),*]) => {
        impl GaussianNoise<$num> {
            pub fn from_diag_sigmas($($args: dtype),*) -> Self {
                let sigmas = Vector::<$num>::new($($args,)*);
                Self::from_vec_sigma(sigmas.as_view())
            }

            pub fn from_diag_covs($($args: dtype,)*) -> Self {
                let sigmas = Vector::<$num>::new($($args,)*);
                Self::from_vec_cov(sigmas.as_view())
            }
        }
    };
}

make_gaussian_vector!(1, [s0]);
make_gaussian_vector!(2, [s0, s1]);
make_gaussian_vector!(3, [s0, s1, s2]);
make_gaussian_vector!(4, [s0, s1, s2, s3]);
make_gaussian_vector!(5, [s0, s1, s2, s3, s4]);
make_gaussian_vector!(6, [s0, s1, s2, s3, s4, s5]);

impl<const N: usize> fmt::Display for GaussianNoise<N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "GaussianNoise{}: {:}", self.dim(), self.sqrt_inf)
    }
}
