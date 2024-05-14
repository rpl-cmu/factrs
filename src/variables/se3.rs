use crate::dtype;
use crate::traits::{LieGroup, Variable};
use crate::variables::{Vector3, VectorD, SO3};
use nalgebra::{dvector, ComplexField, RealField};
use std::fmt;
use std::ops::Mul;

#[derive(Clone, Debug)]
pub struct SE3 {
    rot: SO3,
    xyz: Vector3,
}

impl Variable for SE3 {
    const DIM: usize = 3;

    fn identity() -> Self {
        SE3 {
            rot: SO3::identity(),
            xyz: Vector3::identity(),
        }
    }

    fn inverse(&self) -> Self {
        SE3 {
            rot: self.rot.inverse(),
            xyz: -&self.rot.apply(&self.xyz),
        }
    }

    fn oplus(&self, delta: &VectorD) -> Self {
        let e = Self::exp(delta);
        self * &e
    }

    fn ominus(&self, other: &Self) -> VectorD {
        (&Variable::inverse(self) * other).log()
    }
}

impl LieGroup for SE3 {
    // TODO: Both of this functions need to be tested!
    fn exp(xi: &VectorD) -> Self {
        let xi = dvector![xi[0], xi[1], xi[2]];
        let w = xi.norm();
        let q = SO3::exp(&xi);

        let qv = xi / w;
        let qv = qv * w.sin();
        let qv = qv.push(w.cos());
        let qv = Vector3::new(qv[0], qv[1], qv[2]);

        SE3 { rot: q, xyz: qv }
    }

    fn log(&self) -> VectorD {
        let xi = self.rot.log();
        let w = xi.norm();
        let qv = self.xyz.clone() / w;
        let qv = qv * w.acos();
        dvector![qv[0], qv[1], qv[2]]
    }
}

impl Mul for SE3 {
    type Output = SE3;

    fn mul(self, other: Self) -> SE3 {
        &self * &other
    }
}

impl Mul for &SE3 {
    type Output = SE3;

    fn mul(self, other: Self) -> SE3 {
        SE3 {
            rot: &self.rot * &other.rot,
            xyz: self.rot.apply(&other.xyz) + self.xyz.clone(),
        }
    }
}

impl fmt::Display for SE3 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} {}", self.rot, self.xyz)
    }
}
