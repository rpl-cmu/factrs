// Setup dual num
pub trait DualNum:
    RealField + num_dual::DualNum<dtype> + Into<num_dual::DualVec<dtype, dtype, Dyn>>
{
}
impl<G: RealField + num_dual::DualNum<dtype> + Into<num_dual::DualVec<dtype, dtype, Dyn>>> DualNum
    for G
{
}
pub type DualVec = num_dual::DualVec<dtype, dtype, Dyn>;

use crate::dtype;
use nalgebra as na;

// Re-export all nalgebra types to put default dtype on everything

// Misc imports
pub use na::{dvector, Const, Dyn, RealField};

pub type Vector<const N: usize, D = dtype> = na::SVector<D, N>;
pub type VectorX<D = dtype> = na::DVector<D>;
pub type Vector1<D = dtype> = na::SVector<D, 1>;
pub type Vector2<D = dtype> = na::SVector<D, 2>;
pub type Vector3<D = dtype> = na::SVector<D, 3>;
pub type Vector4<D = dtype> = na::SVector<D, 4>;
pub type Vector5<D = dtype> = na::SVector<D, 5>;
pub type Vector6<D = dtype> = na::SVector<D, 6>;
pub type Vector7<D = dtype> = na::SVector<D, 7>;
pub type Vector8<D = dtype> = na::SVector<D, 8>;
pub type Vector9<D = dtype> = na::SVector<D, 9>;
pub type Vector10<D = dtype> = na::SVector<D, 10>;

// square
pub type MatrixX<D = dtype> = na::DMatrix<D>;
pub type Matrix1<D = dtype> = na::Matrix1<D>;
pub type Matrix2<D = dtype> = na::Matrix2<D>;
pub type Matrix3<D = dtype> = na::Matrix3<D>;
pub type Matrix4<D = dtype> = na::Matrix4<D>;
pub type Matrix5<D = dtype> = na::Matrix5<D>;
pub type Matrix6<D = dtype> = na::Matrix6<D>;

// row
pub type Matrix1xX<D = dtype> = na::Matrix1xX<D>;
pub type Matrix1x2<D = dtype> = na::Matrix1x2<D>;
pub type Matrix1x3<D = dtype> = na::Matrix1x3<D>;
pub type Matrix1x4<D = dtype> = na::Matrix1x4<D>;
pub type Matrix1x5<D = dtype> = na::Matrix1x5<D>;
pub type Matrix1x6<D = dtype> = na::Matrix1x6<D>;

// two rows
pub type Matrix2xX<D = dtype> = na::Matrix2xX<D>;
pub type Matrix2x3<D = dtype> = na::Matrix2x3<D>;
pub type Matrix2x4<D = dtype> = na::Matrix2x4<D>;
pub type Matrix2x5<D = dtype> = na::Matrix2x5<D>;
pub type Matrix2x6<D = dtype> = na::Matrix2x6<D>;

// three rows
pub type Matrix3xX<D = dtype> = na::Matrix3xX<D>;
pub type Matrix3x2<D = dtype> = na::Matrix3x2<D>;
pub type Matrix3x4<D = dtype> = na::Matrix3x4<D>;
pub type Matrix3x5<D = dtype> = na::Matrix3x5<D>;
pub type Matrix3x6<D = dtype> = na::Matrix3x6<D>;

// four rows
pub type Matrix4xX<D = dtype> = na::Matrix4xX<D>;
pub type Matrix4x2<D = dtype> = na::Matrix4x2<D>;
pub type Matrix4x3<D = dtype> = na::Matrix4x3<D>;
pub type Matrix4x5<D = dtype> = na::Matrix4x5<D>;
pub type Matrix4x6<D = dtype> = na::Matrix4x6<D>;

// five rows
pub type Matrix5xX<D = dtype> = na::Matrix5xX<D>;
pub type Matrix5x2<D = dtype> = na::Matrix5x2<D>;
pub type Matrix5x3<D = dtype> = na::Matrix5x3<D>;
pub type Matrix5x4<D = dtype> = na::Matrix5x4<D>;
pub type Matrix5x6<D = dtype> = na::Matrix5x6<D>;

// six rows
pub type Matrix6xX<D = dtype> = na::Matrix6xX<D>;
pub type Matrix6x2<D = dtype> = na::Matrix6x2<D>;
pub type Matrix6x3<D = dtype> = na::Matrix6x3<D>;
pub type Matrix6x4<D = dtype> = na::Matrix6x4<D>;
pub type Matrix6x5<D = dtype> = na::Matrix6x5<D>;

// dynamic rows
pub type MatrixXx2<D = dtype> = na::MatrixXx2<D>;
pub type MatrixXx3<D = dtype> = na::MatrixXx3<D>;
pub type MatrixXx4<D = dtype> = na::MatrixXx4<D>;
pub type MatrixXx5<D = dtype> = na::MatrixXx5<D>;
pub type MatrixXx6<D = dtype> = na::MatrixXx6<D>;
