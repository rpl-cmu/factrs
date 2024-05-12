use crate::variables::DispatchableVariable;
use ahash::AHashMap;
use std::cmp;
use std::collections::hash_map::Entry;
use std::convert::Into;
use std::default::Default;
use std::fmt;
use std::hash;
use std::iter::IntoIterator;
use try_as::traits::TryAsRef;

// We can't add supertraits to DispatchableVariable, so we do it here instead
pub trait Var: DispatchableVariable + fmt::Display + Clone {}
impl<T: DispatchableVariable + fmt::Display + Clone> Var for T {}
pub trait Key: cmp::Eq + cmp::PartialEq + hash::Hash + fmt::Display + Clone {}
impl<T: cmp::Eq + cmp::PartialEq + hash::Hash + fmt::Display + Clone> Key for T {}

#[derive(Clone)]
pub struct Values<K: Key, V: Var> {
    values: AHashMap<K, V>,
}

impl<K: Key, V: Var> Values<K, V> {
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
        V: TryAsRef<T>,
    {
        self.values
            .iter()
            .filter_map(|(_, value)| value.try_as_ref())
    }

    pub fn into_filter<T>(self) -> impl Iterator<Item = T>
    where
        V: TryInto<T>,
    {
        self.values
            .into_iter()
            .filter_map(|(_, value)| value.try_into().ok())
    }
}

impl<K: Key, V: Var> fmt::Display for Values<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if f.alternate() {
            writeln!(f, "{{")?;
            for (key, value) in self.values.iter() {
                writeln!(f, "  {}: {},", key, value)?;
            }
            write!(f, "}}")
        } else {
            write!(f, "{{")?;
            for (key, value) in self.values.iter() {
                write!(f, "{}: {}, ", key, value)?;
            }
            write!(f, "}}")
        }
    }
}

impl<K: Key, V: Var> fmt::Debug for Values<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl<K: Key, V: Var> IntoIterator for Values<K, V> {
    type Item = (K, V);
    type IntoIter = std::collections::hash_map::IntoIter<K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.values.into_iter()
    }
}

impl<K: Key, V: Var> Default for Values<K, V> {
    fn default() -> Self {
        Self {
            values: AHashMap::new(),
        }
    }
}
