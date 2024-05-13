use crate::variables::Variable;
use core::fmt;
use derive_more;
use nalgebra::{DVector, SVector};
use std::ops;

// This is a thin newtype around SVector b/c the way it prints is way too verbose
// This is the only way I've figured out how to change that
#[derive(
    Clone,
    derive_more::Add,
    derive_more::Sub,
    derive_more::Neg,
    derive_more::Div,
    derive_more::Mul,
    derive_more::Deref,
    derive_more::DerefMut,
)]
pub struct Vector<const N: usize>(SVector<f64, N>);

// ------------------------- All ops not in derive_more ------------------------- //
impl<const N: usize> ops::Neg for &Vector<N> {
    type Output = Vector<N>;
    fn neg(self) -> Self::Output {
        Vector(-self.0)
    }
}

impl<const N: usize> fmt::Display for Vector<N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Vector{}(", N)?;
        for i in 0..N - 1 {
            write!(f, "{}, ", self[i])?;
        }
        write!(f, "{})", self[N - 1])
    }
}

impl<const N: usize> fmt::Debug for Vector<N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

// ------------------------- A handful of static functions ------------------------- //
impl<const N: usize> Vector<N> {
    pub fn zeros() -> Self {
        Vector(SVector::zeros())
    }
}

// Modified from: https://docs.rs/nalgebra/latest/src/nalgebra/base/construction.rs.html#960-1103
macro_rules! componentwise_constructors_impl(
    ($($N: expr, [$($($args: ident),*);*] $(;)*)*) => {$(
        impl Vector<$N> {
            #[inline]
            pub const fn new($($($args: f64),*),*) -> Self {
                Vector(SVector::<f64, $N>::new($($($args),*),*))
            }
        }
    )*}
);

componentwise_constructors_impl!(
    1, [x;];
    2, [x; y];
    3, [x; y; z];
    4, [x; y; z; w];
    5, [x; y; z; w; a];
    6, [x; y; z; w; a; b];
);

// ------------------------- Our needs ------------------------- //
impl<const N: usize> Variable for Vector<N> {
    const DIM: usize = N;

    fn identity() -> Self {
        Vector::zeros()
    }

    fn inverse(&self) -> Self {
        -self
    }

    fn oplus(&self, delta: &VectorD) -> Self {
        Vector(self.0 + delta)
    }

    fn ominus(&self, other: &Self) -> VectorD {
        let diff = self.0 - other.0;
        DVector::from_iterator(Self::DIM, diff.iter().cloned())
    }
}

pub type Vector1 = Vector<1>;
pub type Vector2 = Vector<2>;
pub type Vector3 = Vector<3>;
pub type Vector4 = Vector<4>;
pub type Vector5 = Vector<5>;
pub type Vector6 = Vector<6>;
pub type Vector7 = Vector<7>;
pub type Vector8 = Vector<8>;
pub type Vector9 = Vector<9>;
pub type Vector10 = Vector<10>;
pub type VectorD = DVector<f64>;
