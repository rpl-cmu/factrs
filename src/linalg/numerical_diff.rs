use crate::dtype;
use crate::linalg::{Diff, DiffResult, DualScalar, DualVec, MatrixX, VectorX};
use crate::variables::Variable;

use paste::paste;

pub struct NumericalDiff<const PWR: i32 = 6>;

macro_rules! count {
    () => (0usize);
    ( $x:tt $($xs:tt)* ) => (1usize + count!($($xs)*));
}

// ------------------------- Default Implementation (doesn't use duals) ------------------------- //
macro_rules! numerical_maker {
    (grad, $num:expr, $( ($idx:expr, $name:ident, $var:ident) ),*) => {
        paste! {
            #[allow(unused_assignments)]
            pub fn [<gradient_$num>]<$( $var: Variable, )* F: Fn($($var,)*) -> dtype>
                    (f: F, $($name: &$var,)*) -> DiffResult<dtype, VectorX> {
                let eps = dtype::powi(10.0, -PWR);

                // Get Dimension
                let mut dim = 0;
                $(dim += $name.dim();)*
                let num_vars = count!($( $name )*);

                let res = f($( $name.clone(), )*);

                // Compute gradient
                let mut grad: VectorX = VectorX::zeros(dim);
                let mut tvs = [$( VectorX::zeros($name.dim()), )*];

                for i in 0..num_vars {
                    let mut curr_dim = 0;
                    for j in 0..tvs[i].dim() {
                        tvs[i][j] = eps;
                        // TODO: It'd be more efficient to not have to add tangent vectors to each variable
                        // However, I couldn't find a way to do this for a single vector without having to
                        // do a nested iteration of $name which isn't allowed
                        $(let [<$name _og>] = $name.oplus(&tvs[$idx]);)*
                        let plus = f($( [<$name _og>], )*);

                        tvs[i][j] = -eps;
                        $(let [<$name _og>] = $name.oplus(&tvs[$idx]);)*
                        let minus = f($( [<$name _og>], )*);

                        grad[curr_dim + j] = (plus - minus) / (2.0 * eps);
                    }
                    curr_dim += tvs[i].dim();
                }

                DiffResult { value: res, diff: grad }
            }
        }
    };

    (jac, $num:expr, $( ($idx:expr, $name:ident, $var:ident) ),*) => {
        paste! {
            #[allow(unused_assignments)]
            pub fn [<jacobian_$num>]<$( $var: Variable, )* F: Fn($($var,)*) -> VectorX>
                    (f: F, $($name: &$var,)*) -> DiffResult<VectorX, MatrixX> {
                let eps = dtype::powi(10.0, -PWR);

                // Get Dimension
                let mut dim = 0;
                $(dim += $name.dim();)*
                let num_vars = count!($( $name )*);

                let res = f($( $name.clone(), )*);

                // Compute gradient
                let mut jac: MatrixX = MatrixX::zeros(res.len(), dim);
                let mut tvs = [$( VectorX::zeros($name.dim()), )*];

                for i in 0..num_vars {
                    let mut curr_dim = 0;
                    for j in 0..tvs[i].dim() {
                        tvs[i][j] = eps;
                        // TODO: It'd be more efficient to not have to add tangent vectors to each variable
                        // However, I couldn't find a way to do this for a single vector without having to
                        // do a nested iteration of $name which isn't allowed
                        $(let [<$name _og>] = $name.oplus(&tvs[$idx]);)*
                        let plus = f($( [<$name _og>], )*);

                        tvs[i][j] = -eps;
                        $(let [<$name _og>] = $name.oplus(&tvs[$idx]);)*
                        let minus = f($( [<$name _og>], )*);

                        let delta = (plus - minus) / (2.0 * eps);
                        jac.columns_mut(curr_dim + j, 1).copy_from(&delta);
                    }
                    curr_dim += tvs[i].dim();
                }

                DiffResult { value: res, diff: jac }
            }
        }
    };
}

impl<const PWR: i32> NumericalDiff<PWR> {
    pub fn derivative<F: Fn(dtype) -> dtype>(f: F, x: dtype) -> DiffResult<dtype, dtype> {
        let eps = dtype::powi(10.0, -PWR);

        let r = f(x);
        let d = (f(x + eps) - f(x - eps)) / (2.0 * eps);

        DiffResult { value: r, diff: d }
    }

    numerical_maker!(grad, 1, (0, v1, V1));
    numerical_maker!(grad, 2, (0, v1, V1), (1, v2, V2));
    numerical_maker!(grad, 3, (0, v1, V1), (1, v2, V2), (2, v3, V3));
    numerical_maker!(grad, 4, (0, v1, V1), (1, v2, V2), (2, v3, V3), (3, v4, V4));
    numerical_maker!(
        grad,
        5,
        (0, v1, V1),
        (1, v2, V2),
        (2, v3, V3),
        (3, v4, V4),
        (4, v5, V5)
    );
    numerical_maker!(
        grad,
        6,
        (0, v1, V1),
        (1, v2, V2),
        (2, v3, V3),
        (3, v4, V4),
        (4, v5, V5),
        (5, v6, V6)
    );

    numerical_maker!(jac, 1, (0, v1, V1));
    numerical_maker!(jac, 2, (0, v1, V1), (1, v2, V2));
    numerical_maker!(jac, 3, (0, v1, V1), (1, v2, V2), (2, v3, V3));
    numerical_maker!(jac, 4, (0, v1, V1), (1, v2, V2), (2, v3, V3), (3, v4, V4));
    numerical_maker!(
        jac,
        5,
        (0, v1, V1),
        (1, v2, V2),
        (2, v3, V3),
        (3, v4, V4),
        (4, v5, V5)
    );
    numerical_maker!(
        jac,
        6,
        (0, v1, V1),
        (1, v2, V2),
        (2, v3, V3),
        (3, v4, V4),
        (4, v5, V5),
        (5, v6, V6)
    );
}

// ------------------------- Trait Implementation (does use duals) ------------------------- //
macro_rules! numerical_maker_dual {
    (grad, $num:expr, $( ($name:ident, $var:ident) ),*) => {
        paste! {
            #[allow(unused_assignments)]
            fn [<gradient_$num>]<$( $var: Variable, )* F: Fn($($var::Dual,)*) -> DualVec>
                (f: F, $($name: &$var,)*) -> DiffResult<dtype, VectorX>{
                let f_single = |$($name: $var,)*| f($( $name.dual_self(), )*).re;
                Self::[<gradient_$num>](f_single, $($name,)*)
            }
        }
    };

    (jac, $num:expr, $( ($name:ident, $var:ident) ),*) => {
        paste! {
            #[allow(unused_assignments)]
            fn [<jacobian_$num>]<$( $var: Variable, )* F: Fn($($var::Dual,)*) -> VectorX<DualVec>>
                (f: F, $($name: &$var,)*) -> DiffResult<VectorX, MatrixX>{
                let f_single = |$($name: $var,)*| f($( $name.dual_self(), )*).map(|d| d.re);
                Self::[<jacobian_$num>](f_single, $($name,)*)
            }
        }
    };
}

impl<const PWR: i32> Diff for NumericalDiff<PWR> {
    fn derivative<F: Fn(DualScalar) -> DualScalar>(f: F, x: dtype) -> DiffResult<dtype, dtype> {
        let f_single = |x: dtype| f(x.into()).re;
        Self::derivative(f_single, x)
    }

    numerical_maker_dual!(grad, 1, (v1, V1));
    numerical_maker_dual!(grad, 2, (v1, V1), (v2, V2));
    numerical_maker_dual!(grad, 3, (v1, V1), (v2, V2), (v3, V3));
    numerical_maker_dual!(grad, 4, (v1, V1), (v2, V2), (v3, V3), (v4, V4));
    numerical_maker_dual!(grad, 5, (v1, V1), (v2, V2), (v3, V3), (v4, V4), (v5, V5));
    numerical_maker_dual!(
        grad,
        6,
        (v1, V1),
        (v2, V2),
        (v3, V3),
        (v4, V4),
        (v5, V5),
        (v6, V6)
    );

    numerical_maker_dual!(jac, 1, (v1, V1));
    numerical_maker_dual!(jac, 2, (v1, V1), (v2, V2));
    numerical_maker_dual!(jac, 3, (v1, V1), (v2, V2), (v3, V3));
    numerical_maker_dual!(jac, 4, (v1, V1), (v2, V2), (v3, V3), (v4, V4));
    numerical_maker_dual!(jac, 5, (v1, V1), (v2, V2), (v3, V3), (v4, V4), (v5, V5));
    numerical_maker_dual!(
        jac,
        6,
        (v1, V1),
        (v2, V2),
        (v3, V3),
        (v4, V4),
        (v5, V5),
        (v6, V6)
    );
}
