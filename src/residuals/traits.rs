use crate::containers::{Key, Values};
use crate::linalg::{dual_jacobian_1, dual_jacobian_2, dual_jacobian_3, DualVec, MatrixX, VectorX};
use crate::variables::Variable;

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
        let v1: &Self::V1 = unpack(values, &keys[0]);
        dual_jacobian_1(|v1| self.residual1(v1), v1)
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
        let v1: &Self::V1 = unpack(values, &keys[0]);
        let v2: &Self::V2 = unpack(values, &keys[1]);
        dual_jacobian_2(|v1, v2| self.residual2(v1, v2), v1, v2)
    }
}

// ------------------------- Residual 3 ------------------------- //
pub trait Residual3<V: Variable>: Residual<V>
where
    for<'a> &'a V: std::convert::TryInto<&'a Self::V1>,
    for<'a> &'a V: std::convert::TryInto<&'a Self::V2>,
    for<'a> &'a V: std::convert::TryInto<&'a Self::V3>,
{
    type V1: Variable;
    type V2: Variable;
    type V3: Variable;
    const DIM: usize;

    fn residual3(
        &self,
        v1: DualVar<Self::V1>,
        v2: DualVar<Self::V2>,
        v3: DualVar<Self::V3>,
    ) -> VectorX<DualVec>;

    fn residual3_single(&self, v1: Self::V1, v2: &Self::V2, v3: &Self::V3) -> VectorX {
        self.residual3(v1.dual_self(), v2.dual_self(), v3.dual_self())
            .map(|r| r.re)
    }

    fn residual3_jacobian<K: Key>(&self, values: &Values<K, V>, keys: &[K]) -> (VectorX, MatrixX) {
        let v1: &Self::V1 = unpack(values, &keys[0]);
        let v2: &Self::V2 = unpack(values, &keys[1]);
        let v3: &Self::V3 = unpack(values, &keys[2]);
        dual_jacobian_3(|v1, v2, v3| self.residual3(v1, v2, v3), v1, v2, v3)
    }
}
