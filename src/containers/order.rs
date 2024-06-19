use ahash::HashMap;

use crate::variables::Variable;
use std::collections::hash_map::Iter as HashMapIter;

use super::{Key, Values};

// Since the map isn't ordered, we need to track both idx and len of each variable
#[derive(Debug)]
pub struct Idx {
    pub idx: usize,
    pub dim: usize,
}

#[derive(Debug)]
pub struct Order<K: Key> {
    map: HashMap<K, Idx>,
    dim: usize,
}

impl<K: Key> Order<K> {
    pub fn from_values(values: &Values<K, impl Variable>) -> Self {
        let map = values
            .iter()
            .scan(0, |idx, (key, val)| {
                let order = *idx;
                *idx += val.dim();
                Some((
                    key.clone(),
                    Idx {
                        idx: order,
                        dim: val.dim(),
                    },
                ))
            })
            .collect::<HashMap<K, Idx>>();

        let dim = map.values().map(|idx| idx.dim).sum();

        Self { map, dim }
    }

    pub fn get(&self, key: &K) -> Option<&Idx> {
        self.map.get(key)
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn iter(&self) -> HashMapIter<K, Idx> {
        self.map.iter()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::containers::{Symbol, Values, X};
    use crate::variables::{VariableEnum, Vector2, Vector3, Vector6};

    #[test]
    fn new() {
        // Create some form of values
        let mut v: Values<Symbol, VariableEnum> = Values::new();
        v.insert(X(0), Vector2::default());
        v.insert(X(1), Vector6::default());
        v.insert(X(2), Vector3::default());

        // Create an order
        let order = Order::from_values(&v);

        // Verify the order
        assert_eq!(order.len(), 3);
        assert_eq!(order.dim(), 11);
        assert_eq!(order.get(&X(0)).unwrap().dim, 2);
        assert_eq!(order.get(&X(1)).unwrap().dim, 6);
        assert_eq!(order.get(&X(2)).unwrap().dim, 3);
    }
}
