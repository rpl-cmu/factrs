use super::Key;

pub struct Order<K: Key>(Vec<K>);

impl<K: Key> Order<K> {
    pub fn get_items(&self) -> &Vec<K> {
        &self.0
    }
}
