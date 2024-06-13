use crate::{factors::LinearFactor, traits::Key};

pub struct LinearGraph<K: Key> {
    factors: Vec<LinearFactor<K>>,
}

impl<K: Key> LinearGraph<K> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_factor(&mut self, factor: LinearFactor<K>) {
        self.factors.push(factor);
    }

    // pub fn error(&self) -> f64 {
    //     self.factors.iter().map(|f| f.error()).sum()
    // }
}

impl<K: Key> Default for LinearGraph<K> {
    fn default() -> Self {
        Self {
            factors: Vec::new(),
        }
    }
}
