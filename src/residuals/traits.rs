use crate::containers::{Key, Values};
use crate::linalg::{Diff, DualVec, MatrixX, VectorX};
use crate::variables::Variable;
use paste::paste;

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

// ------------------------- Use Macro to create residuals with set sizes ------------------------- //
macro_rules! residual_maker {
    ($num:expr, $( ($idx:expr, $name:ident, $var:ident) ),*) => {
        paste! {
            pub trait [<Residual $num>]<V: Variable>: Residual<V>
            where
                $(
                    for<'a> &'a V: std::convert::TryInto<&'a Self::$var>,
                )*
            {
                $(
                    type $var: Variable;
                )*
                const DIM: usize;
                type Differ: Diff;

                fn [<residual $num>](&self, $($name: DualVar<Self::$var>,)*) -> VectorX<DualVec>;

                fn [<residual $num _single>](&self, $($name: &Self::$var,)*) -> VectorX {
                    self.[<residual $num>]($($name.dual_self(),)*).map(|r| r.re)
                }

                fn [<residual $num _jacobian>]<K: Key>(&self, values: &Values<K, V>, keys: &[K]) -> (VectorX, MatrixX) {
                    // Unwrap everything
                    $(
                        let $name: &Self::$var = unpack(values, &keys[$idx]);
                    )*
                    Self::Differ::[<jacobian_ $num>](|$($name,)*| self.[<residual $num>]($($name,)*), $($name,)*)
                }
            }
        }
    };
}

residual_maker!(1, (0, v1, V1));
residual_maker!(2, (0, v1, V1), (1, v2, V2));
residual_maker!(3, (0, v1, V1), (1, v2, V2), (2, v3, V3));
residual_maker!(4, (0, v1, V1), (1, v2, V2), (2, v3, V3), (3, v4, V4));
residual_maker!(
    5,
    (0, v1, V1),
    (1, v2, V2),
    (2, v3, V3),
    (3, v4, V4),
    (4, v5, V5)
);
residual_maker!(
    6,
    (0, v1, V1),
    (1, v2, V2),
    (2, v3, V3),
    (3, v4, V4),
    (4, v5, V5),
    (5, v6, V6)
);
