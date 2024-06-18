use crate::dtype;
use crate::linalg::{Const, DualScalar, DualVec, Dyn, MatrixX, VectorX};
use crate::variables::Variable;
use paste::paste;

use super::Diff;

pub struct ForwardProp;

macro_rules! forward_maker {
    (grad, $num:expr, $( ($name:ident: $var:ident) ),*) => {
        paste! {
            #[allow(unused_assignments)]
            fn [<gradient_ $num>]<$( $var: Variable, )* F: Fn($($var::Dual,)*) -> DualVec>
                    (f: F, $($name: &$var,)*) -> (dtype, VectorX) {
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

                (res.re, res.eps.unwrap_generic(Dyn(dim), Const::<1>))
            }
        }
    };

    (jac, $num:expr, $( ($name:ident: $var:ident) ),*) => {
        paste! {
            #[allow(unused_assignments)]
            fn [<jacobian_ $num>]<$( $var: Variable, )* F: Fn($($var::Dual,)*) -> VectorX<DualVec>>
                    (f: F, $($name: &$var,)*) -> (VectorX, MatrixX) {
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

                (res.map(|r| r.re), eps)
            }
        }
    };
}

impl Diff for ForwardProp {
    fn derivative<F: Fn(DualScalar) -> DualScalar>(f: F, x: dtype) -> (dtype, dtype) {
        let xd = x.into();
        let r = f(xd);
        (r.re, r.eps)
    }

    forward_maker!(grad, 1, (v1: V1));
    forward_maker!(grad, 2, (v1: V1), (v2: V2));
    forward_maker!(grad, 3, (v1: V1), (v2: V2), (v3: V3));
    forward_maker!(grad, 4, (v1: V1), (v2: V2), (v3: V3), (v4: V4));
    forward_maker!(grad, 5, (v1: V1), (v2: V2), (v3: V3), (v4: V4), (v5: V5));
    forward_maker!(grad, 6, (v1: V1), (v2: V2), (v3: V3), (v4: V4), (v5: V5), (v6: V6));

    forward_maker!(jac, 1, (v1: V1));
    forward_maker!(jac, 2, (v1: V1), (v2: V2));
    forward_maker!(jac, 3, (v1: V1), (v2: V2), (v3: V3));
    forward_maker!(jac, 4, (v1: V1), (v2: V2), (v3: V3), (v4: V4));
    forward_maker!(jac, 5, (v1: V1), (v2: V2), (v3: V3), (v4: V4), (v5: V5));
    forward_maker!(jac, 6, (v1: V1), (v2: V2), (v3: V3), (v4: V4), (v5: V5), (v6: V6));
}
