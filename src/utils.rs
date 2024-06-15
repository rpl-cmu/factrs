use crate::dtype;
use crate::linalg::{MatrixX, VectorX};
use crate::variables::Variable;

// ------------------------- Numerical Derivative (Scalar in/out) ------------------------- //
pub fn num_derivative<F: Fn(dtype) -> dtype>(f: F, x: dtype) -> dtype {
    let eps = 1e-6;

    let x_plus = x + eps;
    let x_minus = x - eps;

    (f(x_plus) - f(x_minus)) / (2.0 * eps)
}

// ------------------------- Numerical Gradient (Vec in / Scalar out) ------------------------- //
pub fn num_gradient<V: Variable, F: Fn(V) -> dtype>(f: F, v: V) -> VectorX {
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
pub fn num_jacobian_11<V: Variable, F: Fn(V) -> VectorX>(f: F, v: V) -> MatrixX {
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
pub fn num_jacobian_21<V1: Variable, V2: Variable, F: Fn(V1, V2) -> VectorX>(
    f: F,
    v1: V1,
    v2: V2,
) -> MatrixX {
    num_jacobian_11(|v| f(v, v2.clone()), v1)
}

pub fn num_jacobian_22<V1: Variable, V2: Variable, F: Fn(V1, V2) -> VectorX>(
    f: F,
    v1: V1,
    v2: V2,
) -> MatrixX {
    num_jacobian_11(|v| f(v1.clone(), v), v2)
}

// ------------------------- Numerical Jacobian - 3 inputs ------------------------- //
pub fn num_jacobian_31<V1: Variable, V2: Variable, V3: Variable, F: Fn(V1, V2, V3) -> VectorX>(
    f: F,
    v1: V1,
    v2: V2,
    v3: V3,
) -> MatrixX {
    num_jacobian_11(|v| f(v, v2.clone(), v3.clone()), v1)
}

pub fn num_jacobian_32<V1: Variable, V2: Variable, V3: Variable, F: Fn(V1, V2, V3) -> VectorX>(
    f: F,
    v1: V1,
    v2: V2,
    v3: V3,
) -> MatrixX {
    num_jacobian_11(|v| f(v1.clone(), v, v3.clone()), v2)
}

pub fn num_jacobian_33<V1: Variable, V2: Variable, V3: Variable, F: Fn(V1, V2, V3) -> VectorX>(
    f: F,
    v1: V1,
    v2: V2,
    v3: V3,
) -> MatrixX {
    num_jacobian_11(|v| f(v1.clone(), v2.clone(), v), v3)
}
