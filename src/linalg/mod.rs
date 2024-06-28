use crate::dtype;
use nalgebra::{self as na, allocator::Allocator};

mod dual;
pub use dual::{DTypes, DualScalar, DualVector, DualVectorX, Numeric};

// Re-export all nalgebra types to put default dtype on everything
// Misc imports
pub use na::{dmatrix, dvector, Const, Dim, DimName, Dyn, RealField};

// Dual numbers
pub use num_dual::Derivative;

// ------------------------- Vector/Matrix Aliases ------------------------- //
// Vectors
pub type Vector<const N: usize, D = dtype> = na::SVector<D, N>;
pub type VectorX<D = dtype> = na::DVector<D>;
pub type Vector1<D = dtype> = na::SVector<D, 1>;
pub type Vector2<D = dtype> = na::SVector<D, 2>;
pub type Vector3<D = dtype> = na::SVector<D, 3>;
pub type Vector4<D = dtype> = na::SVector<D, 4>;
pub type Vector5<D = dtype> = na::SVector<D, 5>;
pub type Vector6<D = dtype> = na::SVector<D, 6>;

pub type VectorView<'a, const N: usize, D = dtype> = na::VectorView<'a, D, Const<N>>;
pub type VectorViewX<'a, D = dtype> = na::VectorView<'a, D, Dyn>;
pub type VectorView1<'a, D = dtype> = na::VectorView<'a, D, Const<1>>;
pub type VectorView2<'a, D = dtype> = na::VectorView<'a, D, Const<2>>;
pub type VectorView3<'a, D = dtype> = na::VectorView<'a, D, Const<3>>;
pub type VectorView4<'a, D = dtype> = na::VectorView<'a, D, Const<4>>;
pub type VectorView5<'a, D = dtype> = na::VectorView<'a, D, Const<5>>;
pub type VectorView6<'a, D = dtype> = na::VectorView<'a, D, Const<6>>;

// Matrices
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

// Views - aka references of matrices
pub type MatrixViewX<'a, D = dtype> = na::MatrixView<'a, D, Dyn, Dyn>;

pub type Matrix<const R: usize, const C: usize = 1, D = dtype> = na::Matrix<
    D,
    Const<R>,
    Const<C>,
    <na::DefaultAllocator as Allocator<D, Const<R>, Const<C>>>::Buffer,
>;
pub type MatrixView<'a, const R: usize, const C: usize = 1, D = dtype> =
    na::MatrixView<'a, D, Const<R>, Const<C>>;

// Generic, taking in sizes with Const
pub type MatrixDim<R, C = Const<1>, D = dtype> =
    na::Matrix<D, R, C, <na::DefaultAllocator as Allocator<D, R, C>>::Buffer>;
pub type MatrixViewDim<'a, R, C = Const<1>, D = dtype> = na::MatrixView<'a, D, R, C>;

// ------------------------- MatrixBlocks ------------------------- //
#[derive(Debug, Clone)]
pub struct MatrixBlock {
    mat: MatrixX,
    idx: Vec<usize>,
}

impl MatrixBlock {
    pub fn new(mat: MatrixX, idx: Vec<usize>) -> Self {
        Self { mat, idx }
    }

    pub fn get_block(&self, idx: usize) -> MatrixViewX<'_> {
        let idx_start = self.idx[idx];
        let idx_end = if idx + 1 < self.idx.len() {
            self.idx[idx + 1]
        } else {
            self.mat.ncols()
        };
        self.mat.columns(idx_start, idx_end - idx_start)
    }

    pub fn mul(&self, idx: usize, x: VectorViewX<'_>) -> VectorX {
        self.get_block(idx) * x
    }

    pub fn mat(&self) -> MatrixViewX<'_> {
        self.mat.as_view()
    }

    pub fn idx(&self) -> &[usize] {
        &self.idx
    }
}

// ------------------------- Derivatives ------------------------- //
use crate::variables::Variable;
use paste::paste;

pub struct DiffResult<V, G> {
    pub value: V,
    pub diff: G,
}

macro_rules! fn_maker {
    (grad, $num:expr, $( ($name:ident: $var:ident) ),*) => {
        paste! {
            fn [<gradient_ $num>]<$( $var: Variable, )* F: Fn($($var::Alias<DualVectorX>,)*) -> DualVectorX>
                    (f: F, $($name: &$var,)*) -> DiffResult<dtype, VectorX>;
        }
    };

    (jac, $num:expr, $( ($name:ident: $var:ident) ),*) => {
        paste! {
            fn [<jacobian_ $num>]<$( $var: Variable, )* F: Fn($($var::Alias<DualVectorX>,)*) -> VectorX<DualVectorX>>
                    (f: F, $($name: &$var,)*) -> DiffResult<VectorX, MatrixX>;
        }
    };
}

pub trait Diff {
    fn derivative<F: Fn(DualScalar) -> DualScalar>(f: F, x: dtype) -> DiffResult<dtype, dtype>;

    // TODO: Default implementation of gradient that calls jacobian with a small wrapper?
    fn_maker!(grad, 1, (v1: V1));
    fn_maker!(grad, 2, (v1: V1), (v2: V2));
    fn_maker!(grad, 3, (v1: V1), (v2: V2), (v3: V3));
    fn_maker!(grad, 4, (v1: V1), (v2: V2), (v3: V3), (v4: V4));
    fn_maker!(grad, 5, (v1: V1), (v2: V2), (v3: V3), (v4: V4), (v5: V5));
    fn_maker!(grad, 6, (v1: V1), (v2: V2), (v3: V3), (v4: V4), (v5: V5), (v6: V6));

    fn_maker!(jac, 1, (v1: V1));
    fn_maker!(jac, 2, (v1: V1), (v2: V2));
    fn_maker!(jac, 3, (v1: V1), (v2: V2), (v3: V3));
    fn_maker!(jac, 4, (v1: V1), (v2: V2), (v3: V3), (v4: V4));
    fn_maker!(jac, 5, (v1: V1), (v2: V2), (v3: V3), (v4: V4), (v5: V5));
    fn_maker!(jac, 6, (v1: V1), (v2: V2), (v3: V3), (v4: V4), (v5: V5), (v6: V6));
}

mod numerical_diff;
pub use numerical_diff::NumericalDiff;

mod forward_prop;
pub use forward_prop::ForwardProp;
