// Similar to gtsam: https://github.com/borglab/gtsam/blob/develop/gtsam/inference/Symbol.cpp
use std::{
    fmt::{self},
    mem::size_of,
};

use crate::prelude::VariableUmbrella;

// Char is stored in last CHR_BITS
// Value is stored in the first IDX_BITS
const KEY_BITS: usize = u64::BITS as usize;
const CHR_BITS: usize = size_of::<char>() * 8;
const IDX_BITS: usize = KEY_BITS - CHR_BITS;
const CHR_MASK: u64 = (char::MAX as u64) << IDX_BITS;
const IDX_MASK: u64 = !CHR_MASK;

// ------------------------- Symbol Parser ------------------------- //

/// Newtype wrap around u64
///
/// In implementation, the u64 is exclusively used, the chr/idx aren't at all.
/// If you'd like to use a custom symbol (ie with two chars for multi-robot
/// experiments), simply define a new trait that creates the u64 as you desire.
#[derive(Clone, Copy, Eq, Hash, PartialEq, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Key(pub u64);

impl Symbol for Key {}

/// This provides a custom conversion two and from a u64 key.
pub trait Symbol: fmt::Debug + Into<Key> {}

pub struct DefaultSymbol {
    chr: char,
    idx: u64,
}

impl DefaultSymbol {
    pub fn new(chr: char, idx: u64) -> Self {
        Self { chr, idx }
    }
}

impl Symbol for DefaultSymbol {}

impl From<Key> for DefaultSymbol {
    fn from(key: Key) -> Self {
        let chr = ((key.0 & CHR_MASK) >> IDX_BITS) as u8 as char;
        let idx = key.0 & IDX_MASK;
        Self { chr, idx }
    }
}

impl From<DefaultSymbol> for Key {
    fn from(sym: DefaultSymbol) -> Key {
        Key((sym.chr as u64) << IDX_BITS | sym.idx & IDX_MASK)
    }
}

impl fmt::Debug for DefaultSymbol {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({}, {})", self.chr, self.idx)
    }
}

// ------------------------- Basic Keys ------------------------- //
/*
To figure out
- How to store in Values and still be print when debugging
    - Probably still want store as a Symbol for this reason

Just want to replace the above functions with some sort of typing

Do the reverse and assign a letter to the variable?
- Not the best in case we have multiple letters for a single variable
*/

pub trait TypedSymbol<V: VariableUmbrella>: Symbol {}

/// Creates and assigns symbols to variables
///
/// To reduce runtime errors, fact.rs symbols are tagged
/// with the type they will be used with. This macro will create a new symbol
/// and implement all the necessary traits for it to be used as a symbol.
/// ```
/// use factrs::prelude::*;
/// assign_symbols!(X: SO2; Y: SE2);
/// ```
#[macro_export]
macro_rules! assign_symbols {
    ($($name:ident : $($var:ident),+);* $(;)?) => {$(
        assign_symbols!($name);

        $(
            impl $crate::containers::TypedSymbol<$var> for $name {}
        )*
    )*};

    ($($name:ident),*) => {
        $(
            #[derive(Clone, Copy)]
            pub struct $name(pub u64);

            paste::paste! {
                impl From<$name> for $crate::containers::DefaultSymbol {
                    fn from(key: $name) -> $crate::containers::DefaultSymbol {
                        let chr = stringify!([<$name:lower>]).chars().next().unwrap();
                        let idx = key.0;
                        $crate::containers::DefaultSymbol::new(chr, idx)
                    }
                }
            }

            impl From<$name> for $crate::containers::Key {
                fn from(key: $name) -> $crate::containers::Key {
                    $crate::containers::DefaultSymbol::from(key).into()
                }
            }

            impl std::fmt::Debug for $name {
                fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                    $crate::containers::DefaultSymbol::from(*self).fmt(f)
                }
            }

            impl $crate::containers::Symbol for $name {}
        )*
    };
}
