// Similar to gtsam: https://github.com/borglab/gtsam/blob/develop/gtsam/inference/Symbol.cpp
use std::{fmt, mem::size_of};

// Char is stored in last CHR_BITS
// Value is stored in the first IDX_BITS
const KEY_BITS: usize = u64::BITS as usize;
const CHR_BITS: usize = size_of::<char>() * 8;
const IDX_BITS: usize = KEY_BITS - CHR_BITS;
const CHR_MASK: u64 = (char::MAX as u64) << IDX_BITS;
const IDX_MASK: u64 = !CHR_MASK;

/// Newtype wrap around u64
/// 
/// First bits contain the index, last bits contain the character.
/// Helpers exist (such as [X], [B], [L]) to create new versions. 
/// 
/// In implementation, the u64 is exclusively used, the chr/idx aren't at all.
/// If you'd like to use a custom symbol (ie with two chars for multi-robot experiments), 
/// simply define a new trait that parses the u64 as you desire to create a new Symbol.
#[derive(Clone, Eq, Hash, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Symbol(u64);

impl Symbol {
    pub fn new_raw(key: u64) -> Self {
        Symbol(key)
    }

    pub fn chr(&self) -> char {
        ((self.0 & CHR_MASK) >> IDX_BITS) as u8 as char
    }

    pub fn idx(&self) -> u64 {
        self.0 & IDX_MASK
    }

    pub fn new(c: char, i: u64) -> Self {
        Symbol(((c as u64) << IDX_BITS) | (i & IDX_MASK))
    }
}

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}{}", self.chr(), self.idx())
    }
}

impl fmt::Debug for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}{}", self.chr(), self.idx())
    }
}

// ------------------------- Helpers ------------------------- //
#[rustfmt::skip]
#[allow(non_snake_case)]
pub fn A(i: u64) -> Symbol { Symbol::new('a', i) }
#[rustfmt::skip]
#[allow(non_snake_case)]
pub fn B(i: u64) -> Symbol { Symbol::new('b', i) }
#[rustfmt::skip]
#[allow(non_snake_case)]
pub fn C(i: u64) -> Symbol { Symbol::new('c', i) }
#[rustfmt::skip]
#[allow(non_snake_case)]
pub fn D(i: u64) -> Symbol { Symbol::new('d', i) }
#[rustfmt::skip]
#[allow(non_snake_case)]
pub fn E(i: u64) -> Symbol { Symbol::new('e', i) }
#[rustfmt::skip]
#[allow(non_snake_case)]
pub fn F(i: u64) -> Symbol { Symbol::new('f', i) }
#[rustfmt::skip]
#[allow(non_snake_case)]
pub fn G(i: u64) -> Symbol { Symbol::new('g', i) }
#[rustfmt::skip]
#[allow(non_snake_case)]
pub fn H(i: u64) -> Symbol { Symbol::new('h', i) }
#[rustfmt::skip]
#[allow(non_snake_case)]
pub fn I(i: u64) -> Symbol { Symbol::new('i', i) }
#[rustfmt::skip]
#[allow(non_snake_case)]
pub fn J(i: u64) -> Symbol { Symbol::new('j', i) }
#[rustfmt::skip]
#[allow(non_snake_case)]
pub fn K(i: u64) -> Symbol { Symbol::new('k', i) }
#[rustfmt::skip]
#[allow(non_snake_case)]
pub fn L(i: u64) -> Symbol { Symbol::new('l', i) }
#[rustfmt::skip]
#[allow(non_snake_case)]
pub fn M(i: u64) -> Symbol { Symbol::new('m', i) }
#[rustfmt::skip]
#[allow(non_snake_case)]
pub fn N(i: u64) -> Symbol { Symbol::new('n', i) }
#[rustfmt::skip]
#[allow(non_snake_case)]
pub fn O(i: u64) -> Symbol { Symbol::new('o', i) }
#[rustfmt::skip]
#[allow(non_snake_case)]
pub fn P(i: u64) -> Symbol { Symbol::new('p', i) }
#[rustfmt::skip]
#[allow(non_snake_case)]
pub fn Q(i: u64) -> Symbol { Symbol::new('q', i) }
#[rustfmt::skip]
#[allow(non_snake_case)]
pub fn R(i: u64) -> Symbol { Symbol::new('r', i) }
#[rustfmt::skip]
#[allow(non_snake_case)]
pub fn S(i: u64) -> Symbol { Symbol::new('s', i) }
#[rustfmt::skip]
#[allow(non_snake_case)]
pub fn T(i: u64) -> Symbol { Symbol::new('t', i) }
#[rustfmt::skip]
#[allow(non_snake_case)]
pub fn U(i: u64) -> Symbol { Symbol::new('u', i) }
#[rustfmt::skip]
#[allow(non_snake_case)]
pub fn V(i: u64) -> Symbol { Symbol::new('v', i) }
#[rustfmt::skip]
#[allow(non_snake_case)]
pub fn W(i: u64) -> Symbol { Symbol::new('w', i) }
#[rustfmt::skip]
#[allow(non_snake_case)]
pub fn X(i: u64) -> Symbol { Symbol::new('x', i) }
#[rustfmt::skip]
#[allow(non_snake_case)]
pub fn Y(i: u64) -> Symbol { Symbol::new('y', i) }
#[rustfmt::skip]
#[allow(non_snake_case)]
pub fn Z(i: u64) -> Symbol { Symbol::new('z', i) }
