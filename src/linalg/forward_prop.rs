use crate::{
    linalg::{Const, DefaultAllocator, DiffResult, DimName, Dyn, MatrixX, VectorDim, VectorX},
    variables::Variable,
};
use paste::paste;

use super::{
    dual::{DualAllocator, DualVector},
    AllocatorBuffer, Diff, MatrixDim,
};

pub struct ForwardProp<N: DimName> {
    _phantom: std::marker::PhantomData<N>,
}

macro_rules! forward_maker {
    ($num:expr, $( ($name:ident: $var:ident) ),*) => {
        paste! {
            #[allow(unused_assignments)]
            fn [<jacobian_ $num>]<$( $var: Variable<Alias<f64> = $var>, )* F: Fn($($var::Alias<Self::D>,)*) -> VectorX<Self::D>>
                    (f: F, $($name: &$var,)*) -> DiffResult<VectorX, MatrixX>{
                // Prepare variables
                let mut curr_dim = 0;
                $(
                    let $name: $var::Alias<Self::D> = $var::dual($name, curr_dim);
                    curr_dim += $name.dim();
                )*

                // Compute residual
                let res = f($($name,)*);

                // Compute Jacobian
                let n = VectorDim::<N>::zeros().shape_generic().0;
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
    };
}

impl<N: DimName> Diff for ForwardProp<N>
where
    AllocatorBuffer<N>: Sync + Send,
    DefaultAllocator: DualAllocator<N>,
    DualVector<N>: Copy,
{
    type D = DualVector<N>;

    forward_maker!(1, (v1: V1));
    forward_maker!(2, (v1: V1), (v2: V2));
    forward_maker!(3, (v1: V1), (v2: V2), (v3: V3));
    forward_maker!(4, (v1: V1), (v2: V2), (v3: V3), (v4: V4));
    forward_maker!(5, (v1: V1), (v2: V2), (v3: V3), (v4: V4), (v5: V5));
    forward_maker!(6, (v1: V1), (v2: V2), (v3: V3), (v4: V4), (v5: V5), (v6: V6));
}
