use crate::dtype;
use crate::linalg::{Const, DualScalar, DualVec, Dyn, MatrixX, VectorX};
use crate::variables::Variable;

// ------------------------- Numerical Derivative (Scalar in/out) ------------------------- //
pub fn dual_derivative<F: Fn(DualScalar) -> DualScalar>(f: F, x: dtype) -> (dtype, dtype) {
    let xd = x.into();
    let r = f(xd);
    (r.re, r.eps)
}

// ------------------------- Numerical Gradient (Vec in / Scalar out) ------------------------- //
pub fn dual_gradient<V: Variable, F: Fn(V::Dual) -> DualVec>(f: F, v: V) -> (dtype, VectorX) {
    let dim = v.dim();
    let vd = v.dual(0, dim);
    let r = f(vd);
    (r.re, r.eps.unwrap_generic(Dyn(dim), Const::<1>))
}

// ------------------------- Numerical Jacobian - 1 input ------------------------- //
pub fn dual_jacobian_1<V: Variable, F: Fn(V::Dual) -> VectorX<DualVec>>(
    f: F,
    v: &V,
) -> (VectorX, MatrixX) {
    // Prepare variables
    let dim = v.dim();
    let v1d = v.dual(0, dim);

    // Compute function value
    let r = f(v1d);

    // Compute Jacobian
    let eps = MatrixX::from_rows(
        r.map(|r| r.eps.unwrap_generic(Dyn(dim), Const::<1>).transpose())
            .as_slice(),
    );

    (r.map(|r| r.re), eps)
}

// ------------------------- Numerical Jacobian - 2 inputs ------------------------- //
pub fn dual_jacobian_2<
    V1: Variable,
    V2: Variable,
    F: Fn(V1::Dual, V2::Dual) -> VectorX<DualVec>,
>(
    f: F,
    v1: &V1,
    v2: &V2,
) -> (VectorX, MatrixX) {
    // Prepare variables
    let dim = V1::DIM + V2::DIM;
    let v1d = v1.dual(0, dim);
    let v2d = v2.dual(V1::DIM, dim);

    // Compute residual
    let res = f(v1d, v2d);

    // Compute Jacobian
    let eps = MatrixX::from_rows(
        res.map(|r| r.eps.unwrap_generic(Dyn(dim), Const::<1>).transpose())
            .as_slice(),
    );

    (res.map(|r| r.re), eps)
}

// ------------------------- Numerical Jacobian - 3 inputs ------------------------- //
pub fn dual_jacobian_3<
    V1: Variable,
    V2: Variable,
    V3: Variable,
    F: Fn(V1::Dual, V2::Dual, V3::Dual) -> VectorX<DualVec>,
>(
    f: F,
    v1: &V1,
    v2: &V2,
    v3: &V3,
) -> (VectorX, MatrixX) {
    // Prepare variables
    let dim = V1::DIM + V2::DIM + V3::DIM;
    let v1d = v1.dual(0, dim);
    let v2d = v2.dual(V1::DIM, dim);
    let v3d = v3.dual(V1::DIM + V2::DIM, dim);

    // Compute residual
    let res = f(v1d, v2d, v3d);

    // Compute Jacobian
    let eps = MatrixX::from_rows(
        res.map(|r| r.eps.unwrap_generic(Dyn(dim), Const::<1>).transpose())
            .as_slice(),
    );

    (res.map(|r| r.re), eps)
}
