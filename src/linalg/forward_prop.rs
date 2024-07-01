use crate::{
    containers::O,
    dtype,
    linalg::{Const, DiffResult, DualScalar, DualVectorX, Dyn, MatrixX, VectorX},
    variables::Variable,
};
use nalgebra::{
    allocator::Allocator,
    constraint::{DimEq, ShapeConstraint},
    DefaultAllocator, Dim, DimAdd, DimName, DimSum, OVector,
};
use paste::paste;

use super::{
    dual::{DualAllocator, DualVectorGeneric},
    Diff, DualVector, Matrix, MatrixDim, MatrixXxN,
};

pub struct ForwardProp;

macro_rules! forward_maker {
    (grad, $num:expr, $( ($name:ident: $var:ident) ),*) => {
        paste! {
            #[allow(unused_assignments)]
            fn [<gradient_ $num>]<$( $var: Variable, )* F: Fn($($var::Alias<DualVectorX>,)*) -> DualVectorX>
                    (f: F, $($name: &$var,)*) -> DiffResult<dtype, VectorX>{
                // Prepare variables
                let mut dim = 0;
                $(
                    dim += $name.dim();
                )*
                let mut curr_dim = 0;
                $(
                    let $name = $name.dual(curr_dim, dim);
                    curr_dim += $name.dim();
                )*

                // Compute residual
                let res = f($($name,)*);

                DiffResult {
                    value: res.re,
                    diff: res.eps.unwrap_generic(Dyn(dim), Const::<1>),
                }
            }
        }
    };

    (jac, $num:expr, $( ($name:ident: $var:ident) ),*) => {
        paste! {
            #[allow(unused_assignments)]
            fn [<jacobian_ $num>]<$( $var: Variable, )* F: Fn($($var::Alias<DualVectorX>,)*) -> VectorX<DualVectorX>>
                    (f: F, $($name: &$var,)*) -> DiffResult<VectorX, MatrixX>{
                // Prepare variables
                let mut dim = 0;
                $(
                    dim += $name.dim();
                )*
                let mut curr_dim = 0;
                $(
                    let $name = $name.dual(curr_dim, dim);
                    curr_dim += $name.dim();
                )*

                // Compute residual
                let res = f($($name,)*);

                // Compute Jacobian
                let eps = MatrixX::from_rows(
                    res.map(|r| r.eps.unwrap_generic(Dyn(dim), Const::<1>).transpose())
                        .as_slice(),
                );

                DiffResult {
                    value: res.map(|r| r.re),
                    diff: eps,
                }
            }
        }
    };
}

impl Diff for ForwardProp {
    fn derivative<F: Fn(DualScalar) -> DualScalar>(f: F, x: dtype) -> DiffResult<dtype, dtype> {
        let xd = x.into();
        let r = f(xd);
        DiffResult {
            value: r.re,
            diff: r.eps,
        }
    }

    // forward_maker!(grad, 1, (v1: V1));
    // forward_maker!(grad, 2, (v1: V1), (v2: V2));
    // forward_maker!(grad, 3, (v1: V1), (v2: V2), (v3: V3));
    // forward_maker!(grad, 4, (v1: V1), (v2: V2), (v3: V3), (v4: V4));
    // forward_maker!(grad, 5, (v1: V1), (v2: V2), (v3: V3), (v4: V4), (v5: V5));
    // forward_maker!(grad, 6, (v1: V1), (v2: V2), (v3: V3), (v4: V4), (v5: V5), (v6: V6));

    // forward_maker!(jac, 1, (v1: V1));
    // forward_maker!(jac, 2, (v1: V1), (v2: V2));
    // forward_maker!(jac, 3, (v1: V1), (v2: V2), (v3: V3));
    // forward_maker!(jac, 4, (v1: V1), (v2: V2), (v3: V3), (v4: V4));
    // forward_maker!(jac, 5, (v1: V1), (v2: V2), (v3: V3), (v4: V4), (v5: V5));
    // forward_maker!(jac, 6, (v1: V1), (v2: V2), (v3: V3), (v4: V4), (v5: V5), (v6: V6));

    fn jacobian_1<
        N: DimName,
        V1: Variable<Alias<f64> = V1>,
        F: Fn(V1::Alias<DualVectorGeneric<N>>) -> VectorX<DualVectorGeneric<N>>,
    >(
        f: F,
        v1: &V1,
    ) -> DiffResult<VectorX, MatrixX>
    where
        <DefaultAllocator as Allocator<dtype, N>>::Buffer: Sync + Send,
        DefaultAllocator: DualAllocator<N>,
    {
        // Prep Variable
        let v1: V1::Alias<DualVectorGeneric<N>> = V1::dual(v1, 0);

        // Compute residual
        let res = f(v1);

        // Compute Jacobian
        let n = OVector::<_, N>::zeros().shape_generic().0;
        let eps1 = MatrixDim::<Dyn, N>::from_rows(
            res.map(|r| r.eps.unwrap_generic(n, Const::<1>).transpose())
                .as_slice(),
        );

        let mut eps = MatrixX::zeros(res.len(), N::USIZE);
        eps.copy_from(&eps1);

        DiffResult {
            value: res.map(|r| r.re),
            diff: eps,
        }
    }

    fn jacobian_2<
        N: DimName,
        V1: Variable<Alias<f64> = V1>,
        V2: Variable<Alias<f64> = V2>,
        F: Fn(
            V1::Alias<DualVectorGeneric<N>>,
            V2::Alias<DualVectorGeneric<N>>,
        ) -> VectorX<DualVectorGeneric<N>>,
    >(
        f: F,
        v1: &V1,
        v2: &V2,
    ) -> DiffResult<VectorX, MatrixX>
    where
        <DefaultAllocator as Allocator<dtype, N>>::Buffer: Sync + Send,
        DefaultAllocator: DualAllocator<N>,
    {
        // Prepare variables
        let v1 = V1::dual(v1, 0);
        let v2 = V2::dual(v2, v1.dim());

        // Compute residual
        let res = f(v1, v2);

        // Compute Jacobian
        let n = OVector::<_, N>::zeros().shape_generic().0;
        let eps1 = MatrixDim::<Dyn, N>::from_rows(
            res.map(|r| r.eps.unwrap_generic(n, Const::<1>).transpose())
                .as_slice(),
        );

        let mut eps = MatrixX::zeros(res.len(), N::USIZE);
        eps.copy_from(&eps1);

        DiffResult {
            value: res.map(|r| r.re),
            diff: eps,
        }
    }
}
