use crate::dtype;
use crate::traits::{DualNum, DualVec, Variable};
use core::fmt;
use derive_more;
use nalgebra::{DVector, SVector};
use std::ops;

// This is a thin newtype around SVector b/c the way it prints is way too verbose
// This is the only way I've figured out how to change that
// TODO: Need derive more to work on references for operators as well
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
pub struct Vector<const N: usize, D: DualNum>(pub SVector<D, N>);

// ------------------------- All ops not in derive_more ------------------------- //
// impl<const N: usize> ops::Div<&dtype> for Vector<N> {
//     type Output = Vector<N>;
//     fn div(self, rhs: &dtype) -> Self::Output {
//         Vector(self.0 / rhs)
//     }
// }

impl<const N: usize, D: DualNum> ops::Neg for &Vector<N, D> {
    type Output = Vector<N, D>;
    fn neg(self) -> Self::Output {
        Vector(-self.0.clone())
    }
}

impl<const N: usize, D: DualNum> fmt::Display for Vector<N, D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Vector{}(", N)?;
        for i in 0..N - 1 {
            write!(f, "{}, ", self[i])?;
        }
        write!(f, "{})", self[N - 1])
    }
}

impl<const N: usize, D: DualNum> fmt::Debug for Vector<N, D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

// ------------------------- A handful of static functions ------------------------- //
impl<const N: usize, D: DualNum> Vector<N, D> {
    pub fn zeros() -> Self {
        Vector(SVector::zeros())
    }
}

// Modified from: https://docs.rs/nalgebra/latest/src/nalgebra/base/construction.rs.html#960-1103
macro_rules! componentwise_constructors_impl(
    ($($N: expr, [$($($args: ident),*);*] $(;)*)*) => {$(
        impl<D: DualNum> Vector<$N,D> {
            #[inline]
            pub const fn new($($($args: D),*),*) -> Self {
                Vector(SVector::<D, $N>::new($($($args),*),*))
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
impl<const N: usize, D: DualNum> Variable<D> for Vector<N, D> {
    const DIM: usize = N;
    type Dual = Vector<N, DualVec>;

    fn identity() -> Self {
        Vector::zeros()
    }

    fn inverse(&self) -> Self {
        -self
    }

    fn oplus(&self, delta: &VectorD<D>) -> Self {
        Vector(&self.0 + delta)
    }

    fn ominus(&self, other: &Self) -> VectorD<D> {
        let diff = &self.0 - &other.0;
        DVector::from_iterator(Self::DIM, diff.iter().cloned())
    }

    fn dual_self(&self) -> Self::Dual {
        Vector(self.0.map(|x| x.into()))
    }
}

pub type Vector1<D = dtype> = Vector<1, D>;
pub type Vector2<D = dtype> = Vector<2, D>;
pub type Vector3<D = dtype> = Vector<3, D>;
pub type Vector4<D = dtype> = Vector<4, D>;
pub type Vector5<D = dtype> = Vector<5, D>;
pub type Vector6<D = dtype> = Vector<6, D>;
pub type Vector7<D = dtype> = Vector<7, D>;
pub type Vector8<D = dtype> = Vector<8, D>;
pub type Vector9<D = dtype> = Vector<9, D>;
pub type Vector10<D = dtype> = Vector<10, D>;
pub type VectorD<D = dtype> = DVector<D>;
