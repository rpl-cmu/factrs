use crate::linalg::{MatrixX, VectorX};
use crate::traits::Key;

pub struct LinearFactor<K: Key> {
    keys: Vec<K>,
    a: MatrixX,
    b: VectorX,
}

impl<K: Key> LinearFactor<K> {
    pub fn new(keys: Vec<K>, a: MatrixX, b: VectorX) -> Self {
        Self { keys, a, b }
    }
}
