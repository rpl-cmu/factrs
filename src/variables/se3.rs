use crate::dtype;
use crate::traits::{DualNum, LieGroup, Variable};
use crate::variables::{Vector3, VectorD, SO3};
use nalgebra::{dvector, ComplexField};
use std::fmt;
use std::ops::Mul;

#[derive(Clone, Debug)]
pub struct SE3<D: DualNum<dtype> = dtype> {
    rot: SO3<D>,
    xyz: Vector3<D>,
}

impl<D: DualNum<dtype>> Variable<D> for SE3<D> {
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

    fn oplus(&self, delta: &VectorD<D>) -> Self {
        let e = Self::exp(delta);
        self * &e
    }

    fn ominus(&self, other: &Self) -> VectorD<D> {
        (&Variable::inverse(self) * other).log()
    }
}

impl<D: DualNum<dtype>> LieGroup<D> for SE3<D> {
    // TODO: Both of this functions need to be tested!
    fn exp(xi: &VectorD<D>) -> Self {
        let xi = dvector![xi[0].clone(), xi[1].clone(), xi[2].clone()];
        let w = xi.norm();
        let q = SO3::exp(&xi);

        let qv = xi / w.clone();
        let qv = qv * w.clone().sin();
        let qv = qv.push(w.cos());
        let qv = Vector3::new(qv[0].clone(), qv[1].clone(), qv[2].clone());

        SE3 { rot: q, xyz: qv }
    }

    fn log(&self) -> VectorD<D> {
        let xi = self.rot.log();
        let w = xi.norm();
        let qv = self.xyz.clone() / w.clone();
        let qv = qv * w.acos();
        dvector![qv[0].clone(), qv[1].clone(), qv[2].clone()]
    }
}

impl<D: DualNum<dtype>> Mul for SE3<D> {
    type Output = SE3<D>;

    fn mul(self, other: Self) -> Self::Output {
        &self * &other
    }
}

impl<D: DualNum<dtype>> Mul for &SE3<D> {
    type Output = SE3<D>;

    fn mul(self, other: Self) -> Self::Output {
        SE3 {
            rot: &self.rot * &other.rot,
            xyz: self.rot.apply(&other.xyz) + self.xyz.clone(),
        }
    }
}

impl<D: DualNum<dtype>> fmt::Display for SE3<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} {}", self.rot, self.xyz)
    }
}
