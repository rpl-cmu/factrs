use crate::linear::LinearValues;
use ahash::AHashMap;
use std::{collections::hash_map::Entry, default::Default, fmt, iter::IntoIterator};

use super::Symbol;
use crate::variables::VariableSafe;

// Since we won't be passing dual numbers through any of this,
// we can just use dtype rather than using generics with Numeric

#[derive(Clone)]
pub struct Values {
    values: AHashMap<Symbol, Box<dyn VariableSafe>>,
}

impl Values {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn entry(&mut self, key: Symbol) -> Entry<Symbol, Box<dyn VariableSafe>> {
        self.values.entry(key)
    }

    pub fn insert(
        &mut self,
        key: Symbol,
        value: impl VariableSafe,
    ) -> Option<Box<dyn VariableSafe>> {
        // TODO: Avoid cloning here?
        self.values.insert(key, value.clone_box())
    }

    pub fn get(&self, key: &Symbol) -> Option<&Box<dyn VariableSafe>> {
        self.values.get(key)
    }

    // TODO: This should be some kind of error
    pub fn get_cast<T: VariableSafe>(&self, key: &Symbol) -> Option<&T> {
        self.values
            .get(key)
            .and_then(|value| value.downcast_ref::<T>())
    }

    // TODO: Does this still fail if one is missing?
    // pub fn get_multiple<'a>(&self, keys: impl IntoIterator<Item = &'a Symbol>) -> Option<Vec<&V>>
    // where
    //     Symbol: 'a,
    // {
    //     keys.into_iter().map(|key| self.values.get(key)).collect()
    // }

    pub fn get_mut(&mut self, key: &Symbol) -> Option<&mut Box<dyn VariableSafe>> {
        self.values.get_mut(key)
    }

    pub fn remove(&mut self, key: &Symbol) -> Option<Box<dyn VariableSafe>> {
        self.values.remove(key)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&Symbol, &Box<dyn VariableSafe>)> {
        self.values.iter()
    }

    pub fn filter<'a, T: 'a + VariableSafe>(&'a self) -> impl Iterator<Item = &'a T> {
        self.values
            .iter()
            .filter_map(|(_, value)| value.downcast_ref::<T>())
    }

    pub fn oplus_mut(&mut self, delta: &LinearValues) {
        for (key, value) in delta.iter() {
            if let Some(v) = self.values.get_mut(key) {
                assert!(v.dim() == value.len(), "Dimension mismatch in values oplus",);
                v.oplus_mut(value);
            }
        }
    }
}

impl fmt::Display for Values {
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

impl fmt::Debug for Values {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl IntoIterator for Values {
    type Item = (Symbol, Box<dyn VariableSafe>);
    type IntoIter = std::collections::hash_map::IntoIter<Symbol, Box<dyn VariableSafe>>;

    fn into_iter(self) -> Self::IntoIter {
        self.values.into_iter()
    }
}

impl Default for Values {
    fn default() -> Self {
        Self {
            values: AHashMap::new(),
        }
    }
}
