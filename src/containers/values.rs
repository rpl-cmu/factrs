use std::{collections::hash_map::Entry, default::Default, fmt, iter::IntoIterator};

use ahash::AHashMap;

use super::Symbol;
use crate::{linear::LinearValues, variables::VariableSafe};

// Since we won't be passing dual numbers through any of this,
// we can just use dtype rather than using generics with Numeric

/// Structure to hold the Variables used in the graph.
///
/// Values is essentially a thing wrapper around a Hashmap that maps [Symbol] ->
/// [VariableSafe]. If you'd like to define a custom variable to be used in
/// Values, it must implement [Variable](crate::variables::Variable), and then
/// will implement [VariableSafe] via a blanket implementation.
/// ```
/// # use factrs::prelude::*;
/// let x = SO2::from_theta(0.1);
/// let mut values = Values::new();
/// values.insert(X(0), x);
/// ```
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

    /// Returns an [std::collections::hash_map::Entry] from the underlying
    /// HashMap.
    pub fn entry(&mut self, key: Symbol) -> Entry<Symbol, Box<dyn VariableSafe>> {
        self.values.entry(key)
    }

    pub fn insert(
        &mut self,
        key: Symbol,
        value: impl VariableSafe,
    ) -> Option<Box<dyn VariableSafe>> {
        self.values.insert(key, Box::new(value))
    }

    /// Returns a dynamic VariableSafe.
    ///
    /// If the underlying value is desired, use [Values::get_cast]
    pub fn get(&self, key: &Symbol) -> Option<&dyn VariableSafe> {
        self.values.get(key).map(|f| f.as_ref())
    }

    // TODO: This should be some kind of error
    /// Casts and returns the underlying variable.
    ///
    /// This will return the value if variable is in the graph, and if the cast
    /// is successful. Returns None otherwise.
    /// ```
    /// # use factrs::prelude::*;
    /// # let x = SO2::from_theta(0.1);
    /// # let mut values = Values::new();
    /// # values.insert(X(0), x);
    /// let x_out = values.get_cast::<SO2>(&X(0));
    /// ```
    pub fn get_cast<T: VariableSafe>(&self, key: &Symbol) -> Option<&T> {
        self.values
            .get(key)
            .and_then(|value| value.downcast_ref::<T>())
    }

    // TODO: Does this still fail if one is missing?
    // pub fn get_multiple<'a>(&self, keys: impl IntoIterator<Item = &'a Symbol>) ->
    // Option<Vec<&V>> where
    //     Symbol: 'a,
    // {
    //     keys.into_iter().map(|key| self.values.get(key)).collect()
    // }

    /// Mutable version of [Values::get].
    pub fn get_mut(&mut self, key: &Symbol) -> Option<&mut dyn VariableSafe> {
        self.values.get_mut(key).map(|f| f.as_mut())
    }

    // TODO: This should be some kind of error
    /// Mutable version of [Values::get_cast].
    pub fn get_mut_cast<T: VariableSafe>(&mut self, key: &Symbol) -> Option<&mut T> {
        self.values
            .get_mut(key)
            .and_then(|value| value.downcast_mut::<T>())
    }

    pub fn remove(&mut self, key: &Symbol) -> Option<Box<dyn VariableSafe>> {
        self.values.remove(key)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&Symbol, &Box<dyn VariableSafe>)> {
        self.values.iter()
    }

    /// Returns a iterator of references of all variables of a specific type in
    /// the values.
    ///
    /// ```
    /// # use factrs::prelude::*;
    /// # let mut values = Values::new();
    /// # (0..10).for_each(|i| {values.insert(X(0), SO2::identity());} );
    /// let mine: Vec<&SO2> = values.filter().collect();
    /// ```
    pub fn filter<'a, T: 'a + VariableSafe>(&'a self) -> impl Iterator<Item = &'a T> {
        self.values
            .iter()
            .filter_map(|(_, value)| value.downcast_ref::<T>())
    }

    /// Update variables in place via the
    /// [oplus](crate::variables::Variable::oplus) operation.
    ///
    /// The [LinearValues] need to be setup to have the same keys and each key
    /// must have a variable of the same length.
    pub fn oplus_mut(&mut self, delta: &LinearValues) {
        // TODO: More error checking here
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
