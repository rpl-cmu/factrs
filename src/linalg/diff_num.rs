use crate::dtype;
use crate::linalg::{Diff, DualScalar, DualVec, MatrixX, VectorX};
use crate::variables::Variable;

use paste::paste;

pub struct Numerical<const PWR: i32 = 6>;

macro_rules! numerical_maker {
    (grad, $num:ident, $( ($name:ident: $var:ident) ),*) => {
        paste! {
            #[allow(unused_assignments)]
            fn [<gradient $num>]<$( $var: Variable, )* F: Fn($($var::Dual,)*) -> DualVec>
                    (f: F, $($name: &$var,)*) -> (dtype, VectorX) {
                let eps = (1.0 as dtype).powi(-PWR);

                // Get Dimension
                let mut dim = 0;
                $(
                    dim += $name.dim();
                )*

                // Helper closure
                let f_single = |$($name: $var,)*| f($($name.dual(0, dim),)*).re;
                let res = f_single($( $name.clone(), )*);
                macro_rules! call_f {
                    () => {
                        f_single($( $name.clone(), )*)
                    };
                }

                // Compute gradient
                let mut grad: VectorX = VectorX::zeros(dim);

                let mut curr_dim = 0;
                $(
                    let [<$name _og>] = $name.clone();
                    for j in 0..$name.dim() {
                        let mut tv: VectorX = VectorX::zeros($name.dim());

                        tv[j] = eps;
                        let $name = $name.oplus(&tv);
                        let plus = call_f!();

                        tv[j] = -eps;
                        let $name = $name.oplus(&tv);
                        let minus = f_single($( $name.clone(), )*);

                        let delta = (plus - minus) / (2.0 * eps);
                        grad[curr_dim + j] = delta;
                        let $name = [<$name _og>].clone();
                    }
                    curr_dim += $name.dim();
                )*

                (res, grad)
            }
        }
    };
}

// impl<const PWR: i32> Diff for Numerical<PWR> {
//     fn derivative<F: Fn(DualScalar) -> DualScalar>(f: F, x: dtype) -> (dtype, dtype) {
//         let eps = (1.0 as dtype).powi(-PWR);

//         let f_single = |x| f(DualScalar::new(x, 1.0)).re;
//         let r = f_single(x);
//         let d = (f_single(x + eps) - f_single(x - eps)) / (2.0 * eps);

//         (r, d)
//     }

//     numerical_maker!(grad, _1, (v1: V1));
//     numerical_maker!(grad, _2, (v1: V1), (v2: V2));
//     numerical_maker!(grad, _3, (v1: V1), (v2: V2), (v3: V3));
//     numerical_maker!(grad, _4, (v1: V1), (v2: V2), (v3: V3), (v4: V4));
//     numerical_maker!(grad, _5, (v1: V1), (v2: V2), (v3: V3), (v4: V4), (v5: V5));
//     numerical_maker!(grad, _6, (v1: V1), (v2: V2), (v3: V3), (v4: V4), (v5: V5), (v6: V6));
// }

// ------------------------- Numerical Derivative (Scalar in/out) ------------------------- //
pub fn num_derivative<F: Fn(dtype) -> dtype>(f: F, x: dtype) -> dtype {
    let eps = 1e-6;

    let x_plus = x + eps;
    let x_minus = x - eps;

    (f(x_plus) - f(x_minus)) / (2.0 * eps)
}

// ------------------------- Numerical Gradient (Vec in / Scalar out) ------------------------- //
pub fn num_gradient<V: Variable, F: Fn(V) -> dtype>(f: F, v: &V) -> VectorX {
    let eps = 1e-6;

    // Prepare variables
    let dim = v.dim();

    // Compute gradient
    let mut grad: VectorX = VectorX::zeros(dim);

    for j in 0..dim {
        let mut tv: VectorX = VectorX::zeros(dim);

        tv[j] = eps;
        let v_plus = v.clone().oplus(&tv);

        tv[j] = -eps;
        let v_minus = v.clone().oplus(&tv);

        let delta = (f(v_plus) - f(v_minus)) / (2.0 * eps);

        grad[j] = delta;
    }

    grad
}

// ------------------------- Numerical Jacobian - 1 input ------------------------- //
pub fn num_jacobian_11<V: Variable, F: Fn(V) -> VectorX>(f: F, v: &V) -> MatrixX {
    let eps = 1e-6;

    // Prepare variables
    let dim = v.dim();

    // Compute jacobian
    let mut jac: MatrixX = MatrixX::zeros(f(v.clone()).len(), dim);

    for j in 0..dim {
        let mut tv: VectorX = VectorX::zeros(dim);

        tv[j] = eps;
        let v_plus = v.clone().oplus(&tv);

        tv[j] = -eps;
        let v_minus = v.clone().oplus(&tv);

        let delta = (f(v_plus) - f(v_minus)) / (2.0 * eps);

        jac.columns_mut(j, 1).copy_from(&delta);
    }

    jac
}

// ------------------------- Numerical Jacobian - 2 inputs ------------------------- //
pub fn num_jacobian_2<V1: Variable, V2: Variable, F: Fn(V1, V2) -> VectorX>(
    f: F,
    v1: &V1,
    v2: &V2,
) -> MatrixX {
    let eps = 1e-6;

    // Prepare variables
    let dim1 = v1.dim();
    let dim2 = v2.dim();

    // Compute jacobian
    let mut jac: MatrixX = MatrixX::zeros(f(v1.clone(), v2.clone()).len(), dim1 + dim2);

    for j in 0..dim1 {
        let mut tv: VectorX = VectorX::zeros(dim1);

        tv[j] = eps;
        let v_plus = v1.clone().oplus(&tv);

        tv[j] = -eps;
        let v_minus = v1.clone().oplus(&tv);

        let delta = (f(v_plus.clone(), v2.clone()) - f(v_minus.clone(), v2.clone())) / (2.0 * eps);

        jac.columns_mut(j, 1).copy_from(&delta);
    }

    for j in 0..dim2 {
        let mut tv: VectorX = VectorX::zeros(dim2);

        tv[j] = eps;
        let v_plus = v2.clone().oplus(&tv);

        tv[j] = -eps;
        let v_minus = v2.clone().oplus(&tv);

        let delta = (f(v1.clone(), v_plus.clone()) - f(v1.clone(), v_minus.clone())) / (2.0 * eps);

        jac.columns_mut(dim1 + j, 1).copy_from(&delta);
    }

    jac
}

pub fn num_jacobian_21<V1: Variable, V2: Variable, F: Fn(V1, V2) -> VectorX>(
    f: F,
    v1: &V1,
    v2: &V2,
) -> MatrixX {
    num_jacobian_11(|v| f(v, v2.clone()), v1)
}

pub fn num_jacobian_22<V1: Variable, V2: Variable, F: Fn(V1, V2) -> VectorX>(
    f: F,
    v1: &V1,
    v2: &V2,
) -> MatrixX {
    num_jacobian_11(|v| f(v1.clone(), v), v2)
}

// ------------------------- Numerical Jacobian - 3 inputs ------------------------- //
pub fn num_jacobian_31<V1: Variable, V2: Variable, V3: Variable, F: Fn(V1, V2, V3) -> VectorX>(
    f: F,
    v1: &V1,
    v2: &V2,
    v3: &V3,
) -> MatrixX {
    num_jacobian_11(|v| f(v, v2.clone(), v3.clone()), v1)
}

pub fn num_jacobian_32<V1: Variable, V2: Variable, V3: Variable, F: Fn(V1, V2, V3) -> VectorX>(
    f: F,
    v1: &V1,
    v2: &V2,
    v3: &V3,
) -> MatrixX {
    num_jacobian_11(|v| f(v1.clone(), v, v3.clone()), v2)
}

pub fn num_jacobian_33<V1: Variable, V2: Variable, V3: Variable, F: Fn(V1, V2, V3) -> VectorX>(
    f: F,
    v1: &V1,
    v2: &V2,
    v3: &V3,
) -> MatrixX {
    num_jacobian_11(|v| f(v1.clone(), v2.clone(), v), v3)
}
