use crate::variables::{self, Key, VariableEnum, VariableEnumDispatch};
use ahash::AHashMap;
use std::collections::hash_map::Entry;
use std::convert::Into;
use std::fmt;

#[derive(Clone, Default)]
pub struct Values {
    values: AHashMap<Key, VariableEnum>,
}

impl Values {
    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn entry(&mut self, key: Key) -> Entry<Key, VariableEnum> {
        self.values.entry(key)
    }

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

impl fmt::Debug for Values {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

macro_rules! values_into_vec {
    ($($variant:ident),*) => {
        $(
            impl<'a> Into<Vec<variables::$variant>> for &Values {
                fn into(self) -> Vec<variables::$variant> {
                    self.values
                        .iter()
                        .filter_map(|(_, value)| match value {
                            VariableEnum::$variant(variant) => Some(variant),
                            _ => None,
                        })
                        .cloned()
                        .collect()
                }
            }

            impl<'a> Into<Vec<&'a variables::$variant>> for &'a Values {
                fn into(self) -> Vec<&'a variables::$variant> {
                    self.values
                        .iter()
                        .filter_map(|(_, value)| match value {
                            VariableEnum::$variant(variant) => Some(variant),
                            _ => None,
                        })
                        .collect()
                }
            }
        )*
    };
}

values_into_vec!(
    SO3, SE3, Vector1, Vector2, Vector3, Vector4, Vector5, Vector6, Vector7, Vector8, Vector9,
    Vector10
);
