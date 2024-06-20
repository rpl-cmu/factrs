use crate::{
    containers::{Idx, Key, Order, Values},
    linalg::{VectorViewX, VectorX},
    variables::Variable,
};
use std::collections::hash_map::Iter as HashMapIter;

pub struct LinearValues<K: Key> {
    values: VectorX,
    order: Order<K>,
}

impl<K: Key> LinearValues<K> {
    pub fn zero_from_order(order: Order<K>) -> Self {
        let values = VectorX::zeros(order.dim());
        Self { values, order }
    }

    pub fn from_order_and_values(order: Order<K>, values: VectorX) -> Self {
        assert!(
            values.len() == order.dim(),
            "Values and order must have the same dimension when creating LinearValues"
        );
        Self { values, order }
    }

    pub fn zero_from_values(values: &Values<K, impl Variable>) -> Self {
        let order = Order::from_values(values);
        let values = VectorX::zeros(order.dim());
        Self { values, order }
    }

    pub fn len(&self) -> usize {
        self.order.len()
    }

    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    pub fn dim(&self) -> usize {
        self.values.len()
    }

    fn get_idx(&self, idx: &Idx) -> VectorViewX<'_> {
        self.values.rows(idx.idx, idx.dim)
    }

    pub fn get(&self, key: &K) -> Option<VectorViewX<'_>> {
        let idx = self.order.get(key)?;
        self.get_idx(idx).into()
    }

    pub fn iter(&self) -> Iter<'_, K> {
        Iter {
            values: self,
            idx: self.order.iter(),
        }
    }
}

pub struct Iter<'a, K: Key> {
    values: &'a LinearValues<K>,
    idx: HashMapIter<'a, K, Idx>,
}

impl<'a, K: Key> Iterator for Iter<'a, K> {
    type Item = (&'a K, VectorViewX<'a>);

    fn next(&mut self) -> Option<Self::Item> {
        let n = self.idx.next()?;
        Some((n.0, self.values.get_idx(n.1)))
    }
}

#[cfg(test)]
mod test {
    use crate::{
        containers::{Symbol, X},
        linalg::{Vector2, Vector3, Vector6},
        variables::VariableEnum,
    };

    use super::*;

    fn make_order_vector() -> (Order<Symbol>, VectorX) {
        // Create some form of values
        let mut v: Values<Symbol, VariableEnum> = Values::new();
        v.insert(X(0), Vector2::default());
        v.insert(X(1), Vector6::default());
        v.insert(X(2), Vector3::default());

        // Create an order
        let order = Order::from_values(&v);
        let vector = VectorX::from_fn(order.dim(), |i, _| i as f64);
        (order, vector)
    }

    #[test]
    fn from_order_and_values() {
        let (order, vector) = make_order_vector();

        // Create LinearValues
        let linear_values = LinearValues::from_order_and_values(order, vector);
        assert!(linear_values.len() == 3);
        assert!(linear_values.dim() == 11);
        assert!(linear_values.get(&X(0)).unwrap().len() == 2);
        assert!(linear_values.get(&X(1)).unwrap().len() == 6);
        assert!(linear_values.get(&X(2)).unwrap().len() == 3);
        assert!(linear_values.get(&X(3)).is_none());
    }

    #[test]
    #[should_panic]
    fn mismatched_size() {
        let (order, vector) = make_order_vector();
        let vector = vector.push(0.0);
        LinearValues::from_order_and_values(order, vector);
    }
}