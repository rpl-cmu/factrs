// Key
use std::{cmp, fmt, hash};
pub trait Key: cmp::Eq + cmp::PartialEq + hash::Hash + fmt::Display + fmt::Debug + Clone {}
impl<T: cmp::Eq + cmp::PartialEq + hash::Hash + fmt::Display + fmt::Debug + Clone> Key for T {}

mod symbol;
pub use symbol::*;

mod values;
pub use values::Values;

mod order;
pub use order::{Idx, ValuesOrder};

mod graph;
pub use graph::{Graph, GraphOrder};
