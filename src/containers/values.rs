use crate::{dtype, linalg::VectorX};
use ahash::AHashMap;
use std::collections::hash_map::Entry;
use std::convert::Into;
use std::default::Default;
use std::fmt;
use std::iter::IntoIterator;

use crate::traits::{Key, Variable};

// Since we won't be passing dual numbers through any of this,
// we can just use dtype rather than using generics with DualNum

#[derive(Clone)]
pub struct Values<K: Key, V: Variable<dtype>> {
    values: AHashMap<K, V>,
}

impl<K: Key, V: Variable<dtype>> Values<K, V> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn entry(&mut self, key: K) -> Entry<K, V> {
        self.values.entry(key)
    }

    pub fn insert(&mut self, key: K, value: impl Into<V>) -> Option<V> {
        self.values.insert(key, value.into())
    }

    pub fn get(&self, key: &K) -> Option<&V> {
        self.values.get(key)
    }

    pub fn get_multiple<'a>(&self, keys: impl IntoIterator<Item = &'a K>) -> Option<Vec<&V>>
    where
        K: 'a,
    {
        keys.into_iter().map(|key| self.values.get(key)).collect()
    }

    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        self.values.get_mut(key)
    }

    pub fn remove(&mut self, key: &K) -> Option<V> {
        self.values.remove(key)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.values.iter()
    }

    pub fn filter<'a, T: 'a>(&'a self) -> impl Iterator<Item = &'a T>
    where
        &'a V: TryInto<&'a T>,
    {
        self.values
            .iter()
            .filter_map(|(_, value)| value.try_into().ok())
    }

    pub fn into_filter<T>(self) -> impl Iterator<Item = T>
    where
        V: TryInto<T>,
    {
        self.values
            .into_iter()
            .filter_map(|(_, value)| value.try_into().ok())
    }

    pub fn oplus(&mut self, delta: &VectorValues<K>) {
        for (key, value) in delta.iter() {
            if let Some(v) = self.values.get_mut(key) {
                // TODO: Should we panic here if the dimensions don't match?
                // TODO: Or should we return result in some way?
                assert!(v.dim() == value.dim(), "Dimension mismatch in values oplus",);
                *v = v.oplus(value);
            }
        }
    }
}

impl<K: Key, V: Variable<dtype>> fmt::Display for Values<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if f.alternate() {
            writeln!(f, "{{")?;
            for (key, value) in self.values.iter() {
                writeln!(f, "  {}: {:?},", key, value)?;
            }
            write!(f, "}}")
        } else {
            write!(f, "{{")?;
            for (key, value) in self.values.iter() {
                write!(f, "{}: {:?}, ", key, value)?;
            }
            write!(f, "}}")
        }
    }
}

impl<K: Key, V: Variable<dtype>> fmt::Debug for Values<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl<K: Key, V: Variable<dtype>> IntoIterator for Values<K, V> {
    type Item = (K, V);
    type IntoIter = std::collections::hash_map::IntoIter<K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.values.into_iter()
    }
}

impl<K: Key, V: Variable<dtype>> Default for Values<K, V> {
    fn default() -> Self {
        Self {
            values: AHashMap::new(),
        }
    }
}

// ------------------------- Type Aliases ------------------------- //
pub type VectorValues<K> = Values<K, VectorX>;
