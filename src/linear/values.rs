use std::collections::hash_map::Iter as HashMapIter;

use crate::{
    containers::{Idx, Key, Symbol, Values, ValuesOrder},
    linalg::{VectorViewX, VectorX},
};

pub struct LinearValues {
    values: VectorX,
    order: ValuesOrder,
}

impl LinearValues {
    pub fn zero_from_order(order: ValuesOrder) -> Self {
        let values = VectorX::zeros(order.dim());
        Self { values, order }
    }

    pub fn zero_from_values(values: &Values) -> Self {
        let order = ValuesOrder::from_values(values);
        let values = VectorX::zeros(order.dim());
        Self { values, order }
    }

    pub fn from_order_and_vector(order: ValuesOrder, values: VectorX) -> Self {
        assert!(
            values.len() == order.dim(),
            "Vector and order must have the same dimension when creating LinearValues"
        );
        Self { values, order }
    }

    pub fn from_values_and_vector(values: &Values, vector: VectorX) -> Self {
        let order = ValuesOrder::from_values(values);
        assert!(
            vector.len() == order.dim(),
            "Vector and values must have the same dimension when creating LinearValues"
        );
        Self::from_order_and_vector(order, vector)
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

    pub fn get(&self, key: &Symbol) -> Option<VectorViewX<'_>> {
        let idx = self.order.get(key)?;
        self.get_idx(idx).into()
    }

    pub fn iter(&self) -> Iter<'_, Symbol> {
        Iter {
            values: self,
            idx: self.order.iter(),
        }
    }
}

pub struct Iter<'a, K: Key> {
    values: &'a LinearValues,
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
    use super::*;
    use crate::{
        containers::X,
        dtype,
        variables::{Variable, VectorVar2, VectorVar3, VectorVar6},
    };

    fn make_order_vector() -> (ValuesOrder, VectorX) {
        // Create some form of values
        let mut v = Values::new();
        v.insert(X(0), VectorVar2::identity());
        v.insert(X(1), VectorVar6::identity());
        v.insert(X(2), VectorVar3::identity());

        // Create an order
        let order = ValuesOrder::from_values(&v);
        let vector = VectorX::from_fn(order.dim(), |i, _| i as dtype);
        (order, vector)
    }

    #[test]
    fn from_order_and_vector() {
        let (order, vector) = make_order_vector();

        // Create LinearValues
        let linear_values = LinearValues::from_order_and_vector(order, vector);
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
        LinearValues::from_order_and_vector(order, vector);
    }
}
