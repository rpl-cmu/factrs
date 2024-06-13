use crate::containers::VectorValues;
use crate::linalg::{MatrixX, VectorX};
use crate::traits::Key;

pub struct LinearFactor<K: Key> {
    pub keys: Vec<K>,
    pub a: MatrixX,
    pub b: VectorX,
}
// TODO: Do I actually need a linear factor? Or can we just use Factor with a linear residual?
impl<K: Key> LinearFactor<K> {
    pub fn new(keys: Vec<K>, a: MatrixX, b: VectorX) -> Self {
        Self { keys, a, b }
    }

    // TODO: What kind of container should this take in? Just Values again? Or should we make a custom linear container?
    // pub fn error(&self, values: &VectorX) -> f64 {
    //     let x = self
    //         .keys
    //         .iter()
    //         .map(|k| values[k.index()])
    //         .collect::<VectorX>();
    //     let e = self.a * x - self.b;
    //     e.norm()
    // }

    #[allow(unused_variables)]
    pub fn jacobian<I: IntoIterator<Item = K>>(order: I) -> (MatrixX, VectorX) {
        unimplemented!()
    }

    #[allow(unused_variables)]
    pub fn solve<I: IntoIterator<Item = K>>(order: I) -> VectorValues<K> {
        unimplemented!()
    }
}
