use crate::linalg::{Const, Dyn, MatrixX, VectorX};
use crate::traits::{DualVec, Variable};
use crate::variables::Values;

use crate::traits::Key;

type DualVar<V> = <V as Variable>::Dual;

// ------------------------- Base Residual Trait & Helpers ------------------------- //
pub fn unpack<'a, V, W, K>(values: &'a Values<K, V>, k: &K) -> &'a W
where
    W: Variable,
    V: Variable,
    &'a V: std::convert::TryInto<&'a W>,
    K: Key,
{
    let v1 = values
        .get(k)
        .unwrap_or_else(|| panic!("Key not found: {}", k));

    v1.try_into().unwrap_or_else(|_| {
        panic!(
            "Variable type mismatch: expected {} for key {}",
            std::any::type_name::<W>(),
            k
        )
    })
}

pub trait Residual<V: Variable>: Sized {
    const DIM: usize;

    fn dim(&self) -> usize {
        Self::DIM
    }

    // TODO: Would be nice if this was generic over dtypes, but it'll probably mostly be used with dual vecs
    fn residual<K: Key>(&self, values: &Values<K, V>, keys: &[K]) -> VectorX {
        self.residual_jacobian(values, keys).0
    }

    fn residual_jacobian<K: Key>(&self, values: &Values<K, V>, keys: &[K]) -> (VectorX, MatrixX);
}

// ------------------------- Residual 1 ------------------------- //
pub trait Residual1<V: Variable>: Residual<V>
where
    for<'a> &'a V: std::convert::TryInto<&'a Self::V1>,
{
    type V1: Variable;
    const DIM: usize;

    fn residual1(&self, v: DualVar<Self::V1>) -> VectorX<DualVec>;

    fn residual1_single(&self, v: &Self::V1) -> VectorX {
        self.residual1(v.dual_self()).map(|r| r.re)
    }

    fn residual1_jacobian<K: Key>(&self, values: &Values<K, V>, keys: &[K]) -> (VectorX, MatrixX) {
        // Unwrap everything
        let v1 = unpack(values, &keys[0]);

        // Prepare variables
        let dim = Self::V1::DIM;
        let v1d = v1.dual(0, dim);

        // Compute residual
        let res = self.residual1(v1d);

        // Compute Jacobian
        let eps = MatrixX::from_rows(
            res.map(|r| r.eps.unwrap_generic(Dyn(dim), Const::<1>).transpose())
                .as_slice(),
        );

        (res.map(|r| r.re), eps)
    }
}

// ------------------------- Residual 2 ------------------------- //
pub trait Residual2<V: Variable>: Residual<V>
where
    for<'a> &'a V: std::convert::TryInto<&'a Self::V1>,
    for<'a> &'a V: std::convert::TryInto<&'a Self::V2>,
{
    type V1: Variable;
    type V2: Variable;
    const DIM: usize;

    fn residual2(&self, v1: DualVar<Self::V1>, v2: DualVar<Self::V2>) -> VectorX<DualVec>;

    fn residual1_single(&self, v1: Self::V1, v2: &Self::V2) -> VectorX {
        self.residual2(v1.dual_self(), v2.dual_self()).map(|r| r.re)
    }

    fn residual2_jacobian<K: Key>(&self, values: &Values<K, V>, keys: &[K]) -> (VectorX, MatrixX) {
        // Unwrap everything
        let v1: &Self::V1 = unpack(values, &keys[0]);
        let v2: &Self::V2 = unpack(values, &keys[1]);

        // Prepare variables
        let dim = Self::V1::DIM + Self::V2::DIM;
        let v1d = v1.dual(0, dim);
        let v2d = v2.dual(Self::V1::DIM, dim);

        // Compute residual
        let res = self.residual2(v1d, v2d);

        // Compute Jacobian
        let eps = MatrixX::from_rows(
            res.map(|r| r.eps.unwrap_generic(Dyn(dim), Const::<1>).transpose())
                .as_slice(),
        );

        (res.map(|r| r.re), eps)
    }
}
