use crate::variables::{Key, VariableEnum, VariableEnumDispatch};
use ahash::AHashMap;
use std::collections::hash_map::{Entry, OccupiedEntry, VacantEntry};

pub struct Values {
    values: AHashMap<Key, VariableEnum>,
}

pub struct OccupiedError<'a, K, V> {
    pub entry: OccupiedEntry<'a, K, V>,
}

pub struct VacantError<'a, K, V> {
    pub entry: VacantEntry<'a, K, V>,
}

impl Values {
    pub fn new() -> Self {
        Self {
            values: AHashMap::new(),
        }
    }

    // Possible errors:
    // - key already exists
    pub fn insert<V: VariableEnumDispatch>(
        &mut self,
        key: Key,
        value: V,
    ) -> Result<&mut VariableEnum, OccupiedError<Key, VariableEnum>> {
        let value: VariableEnum = value.into();

        match self.values.entry(key) {
            Entry::Occupied(entry) => Err(OccupiedError { entry }),
            Entry::Vacant(entry) => {
                let v = entry.insert(value);
                Ok(v)
            }
        }
    }

    // Possible errors:
    // - key not found
    // - key found but type mismatch
    // pub fn update<V: VariableEnumDispatch>(
    //     &mut self,
    //     key: Key,
    //     value: V,
    // ) -> Result<V, VacantError<Key, VariableEnum>> {
    //     let value: VariableEnum = value.into();

    //     match self.values.entry(key) {
    //         Entry::Vacant(entry) => Err(VacantError { entry }),
    //         Entry::Occupied(mut entry) => {
    //             let v: V = entry.insert(value).try_into();
    //             Ok(v)
    //         }
    //     }
    // }

    // TODO: Clean up rest of these functions
    // Possible issues:
    // - key not found
    // - key found but type mismatch
    pub fn get<T: VariableEnumDispatch>(&self, key: &Key) -> Result<&T, Error> {
        let val = self.values.get(key);
    }

    // Possible issues:
    // - key not found
    // - key found but type mismatch
    // pub fn get_mut(&mut self, key: &Key) -> Option<&mut VariableEnum> {
    //     self.values.get_mut(key)
    // }

    // Possible issues:
    // - key not found
    // - key found but type mismatch
    // pub fn remove(&mut self, key: &Key) -> Option<VariableEnum> {
    //     self.values.remove(key)
    // }
}
