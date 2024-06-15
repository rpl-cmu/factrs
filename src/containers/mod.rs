// Key
use std::{cmp, fmt, hash};
pub trait Key: cmp::Eq + cmp::PartialEq + hash::Hash + fmt::Display + Clone {}
impl<T: cmp::Eq + cmp::PartialEq + hash::Hash + fmt::Display + Clone> Key for T {}

mod symbol;
pub use symbol::*;

mod values;
pub use values::{Values, VectorValues};

mod order;
pub use order::Order;

mod graph;
pub use graph::Graph;

mod graph_linear;
pub use graph_linear::LinearGraph;
