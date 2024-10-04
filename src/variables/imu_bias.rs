use std::{fmt, ops};

use nalgebra::Const;

use super::Variable;
use crate::{
    dtype,
    linalg::{DualVector, Numeric, Vector3, VectorDim},
    residuals::{Accel, Gyro},
    tag_variable,
};

tag_variable!(ImuBias);

// TODO: Use newtypes internally as well?
/// IMU bias
///
/// The IMU bias is a 6D vector containing the gyro and accel biases. It is
/// treated as a 6D vector for optimization purposes.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ImuBias<D: Numeric = dtype> {
    gyro: Vector3<D>,
    accel: Vector3<D>,
}

impl<D: Numeric> ImuBias<D> {
    /// Create a new IMU bias
    pub fn new(gyro: Gyro<D>, accel: Accel<D>) -> Self {
        ImuBias {
            gyro: gyro.0,
            accel: accel.0,
        }
    }

    /// Create an IMU bias of zeros (same as identity)
    pub fn zeros() -> Self {
        ImuBias {
            gyro: Vector3::zeros(),
            accel: Vector3::zeros(),
        }
    }

    /// Get the gyro bias
    pub fn gyro(&self) -> &Vector3<D> {
        &self.gyro
    }

    /// Get the accel bias
    pub fn accel(&self) -> &Vector3<D> {
        &self.accel
    }
}

impl<D: Numeric> Variable<D> for ImuBias<D> {
    type Dim = crate::linalg::Const<6>;
    type Alias<DD: Numeric> = ImuBias<DD>;

    fn identity() -> Self {
        ImuBias {
            gyro: Vector3::zeros(),
            accel: Vector3::zeros(),
        }
    }

    fn inverse(&self) -> Self {
        ImuBias {
            gyro: -self.gyro,
            accel: -self.accel,
        }
    }

    fn compose(&self, other: &Self) -> Self {
        ImuBias {
            gyro: self.gyro + other.gyro,
            accel: self.accel + other.accel,
        }
    }

    fn exp(delta: crate::linalg::VectorViewX<D>) -> Self {
        let gyro = Vector3::new(delta[0], delta[1], delta[2]);
        let accel = Vector3::new(delta[3], delta[4], delta[5]);
        ImuBias { gyro, accel }
    }

    fn log(&self) -> crate::linalg::VectorX<D> {
        crate::linalg::vectorx![
            self.gyro.x,
            self.gyro.y,
            self.gyro.z,
            self.accel.x,
            self.accel.y,
            self.accel.z
        ]
    }

    fn dual_convert<DD: Numeric>(other: &Self::Alias<dtype>) -> Self::Alias<DD> {
        ImuBias {
            gyro: other.gyro.map(|x| x.into()),
            accel: other.accel.map(|x| x.into()),
        }
    }

    fn dual_setup<N: nalgebra::DimName>(idx: usize) -> Self::Alias<crate::linalg::DualVector<N>>
    where
        crate::linalg::AllocatorBuffer<N>: Sync + Send,
        nalgebra::DefaultAllocator: crate::linalg::DualAllocator<N>,
        crate::linalg::DualVector<N>: Copy,
    {
        let n = VectorDim::<N>::zeros().shape_generic().0;

        let mut gyro = Vector3::<DualVector<N>>::zeros();
        for (i, gi) in gyro.iter_mut().enumerate() {
            gi.eps = num_dual::Derivative::derivative_generic(n, Const::<1>, idx + i);
        }

        let mut accel = Vector3::<DualVector<N>>::zeros();
        for (i, ai) in accel.iter_mut().enumerate() {
            ai.eps = num_dual::Derivative::derivative_generic(n, Const::<1>, idx + i + 3);
        }

        ImuBias { gyro, accel }
    }
}

impl<D: Numeric> fmt::Display for ImuBias<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ImuBias(g: ({:.3}, {:.3}, {:.3}), a: ({:.3}, {:.3}, {:.3}))",
            self.gyro.x, self.gyro.y, self.gyro.z, self.accel.x, self.accel.y, self.accel.z
        )
    }
}

impl<D: Numeric> fmt::Debug for ImuBias<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl<D: Numeric> ops::Sub for ImuBias<D> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        ImuBias {
            gyro: self.gyro - rhs.gyro,
            accel: self.accel - rhs.accel,
        }
    }
}

impl<'a, D: Numeric> ops::Sub for &'a ImuBias<D> {
    type Output = ImuBias<D>;

    fn sub(self, rhs: Self) -> ImuBias<D> {
        ImuBias {
            gyro: self.gyro - rhs.gyro,
            accel: self.accel - rhs.accel,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_variable;

    test_variable!(ImuBias);
}
