use crate::{
    dtype,
    linalg::{Diff, DiffResult, MatrixX, VectorX},
    variables::Variable,
};

use paste::paste;

pub struct NumericalDiff<const PWR: i32 = 6>;

macro_rules! count {
    () => (0usize);
    ( $x:tt $($xs:tt)* ) => (1usize + count!($($xs)*));
}

// ------------------------- Default Implementation (doesn't use duals) ------------------------- //
macro_rules! numerical_maker {
    ($num:expr, $( ($idx:expr, $name:ident, $var:ident) ),*) => {
        paste! {
            #[allow(unused_assignments)]
            fn [<jacobian_$num>]<$( $var: Variable, )* F: Fn($($var,)*) -> VectorX>
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
                    for j in 0..tvs[i].len() {
                        tvs[i][j] = eps;
                        // TODO: It'd be more efficient to not have to add tangent vectors to each variable
                        // However, I couldn't find a way to do this for a single vector without having to
                        // do a nested iteration of $name which isn't allowed
                        $(let [<$name _og>] = $name.oplus(tvs[$idx].as_view());)*
                        let plus = f($( [<$name _og>], )*);

                        tvs[i][j] = -eps;
                        $(let [<$name _og>] = $name.oplus(tvs[$idx].as_view());)*
                        let minus = f($( [<$name _og>], )*);

                        let delta = (plus - minus) / (2.0 * eps);
                        jac.columns_mut(curr_dim + j, 1).copy_from(&delta);
                    }
                    curr_dim += tvs[i].len();
                }

                DiffResult { value: res, diff: jac }
            }
        }
    };
}

impl<const PWR: i32> Diff for NumericalDiff<PWR> {
    type D = dtype;

    numerical_maker!(1, (0, v1, V1));
    numerical_maker!(2, (0, v1, V1), (1, v2, V2));
    numerical_maker!(3, (0, v1, V1), (1, v2, V2), (2, v3, V3));
    numerical_maker!(4, (0, v1, V1), (1, v2, V2), (2, v3, V3), (3, v4, V4));
    numerical_maker!(
        5,
        (0, v1, V1),
        (1, v2, V2),
        (2, v3, V3),
        (3, v4, V4),
        (4, v5, V5)
    );
    numerical_maker!(
        6,
        (0, v1, V1),
        (1, v2, V2),
        (2, v3, V3),
        (3, v4, V4),
        (4, v5, V5),
        (5, v6, V6)
    );
}
