use crate::variables::{Key, VariableEnum, VariableEnumDispatch};
use ahash::AHashMap;
use std::collections::hash_map::Entry;
use std::fmt;

pub struct Values {
    values: AHashMap<Key, VariableEnum>,
}

impl Values {
    pub fn new() -> Self {
        Self {
            values: AHashMap::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn entry(&mut self, key: Key) -> Entry<Key, VariableEnum> {
        self.values.entry(key)
    }

    // Possible errors:
    // - key already exists
    pub fn insert<V: VariableEnumDispatch>(&mut self, key: Key, value: V) -> Option<VariableEnum> {
        let value: VariableEnum = value.into();
        self.values.insert(key, value)
    }

    pub fn get(&self, key: &Key) -> Option<&VariableEnum> {
        self.values.get(key)
    }

    pub fn get_mut(&mut self, key: &Key) -> Option<&mut VariableEnum> {
        self.values.get_mut(key)
    }

    pub fn remove(&mut self, key: &Key) -> Option<VariableEnum> {
        self.values.remove(key)
    }
}

impl fmt::Display for Values {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if f.alternate() {
            write!(f, "{{\n")?;
            for (key, value) in self.values.iter() {
                write!(f, "  {}: {},\n", key, value)?;
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

impl fmt::Debug for Values {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}
