// Similar to gtsam: https://github.com/borglab/gtsam/blob/develop/gtsam/inference/Key.cpp
use std::fmt;
use std::mem::size_of;

// Char is stored in last CHR_BITS
// Value is stored in the first IDX_BITS
const KEY_BITS: usize = size_of::<u64>() * 8;
const CHR_BITS: usize = size_of::<char>() * 8;
const IDX_BITS: usize = KEY_BITS - CHR_BITS;
const CHR_MASK: u64 = (char::MAX as u64) << IDX_BITS;
const IDX_MASK: u64 = !CHR_MASK;

#[derive(Clone, Eq, Hash, PartialEq)]
pub struct Key(u64);

impl Key {
    pub fn chr(&self) -> char {
        ((self.0 & CHR_MASK) >> IDX_BITS) as u8 as char
    }

    pub fn idx(&self) -> u64 {
        self.0 & IDX_MASK
    }

    pub fn new(c: char, i: u64) -> Self {
        Key(((c as u64) << IDX_BITS) | (i & IDX_MASK))
    }
}

impl fmt::Display for Key {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}{}", self.chr(), self.idx())
    }
}

// ------------------------- Helpers ------------------------- //
#[rustfmt::skip]
#[allow(non_snake_case)]
pub fn A(i: u64) -> Key { Key::new('a', i) }
#[rustfmt::skip]
#[allow(non_snake_case)]
pub fn B(i: u64) -> Key { Key::new('b', i) }
#[rustfmt::skip]
#[allow(non_snake_case)]
pub fn C(i: u64) -> Key { Key::new('c', i) }
#[rustfmt::skip]
#[allow(non_snake_case)]
pub fn D(i: u64) -> Key { Key::new('d', i) }
#[rustfmt::skip]
#[allow(non_snake_case)]
pub fn E(i: u64) -> Key { Key::new('e', i) }
#[rustfmt::skip]
#[allow(non_snake_case)]
pub fn F(i: u64) -> Key { Key::new('f', i) }
#[rustfmt::skip]
#[allow(non_snake_case)]
pub fn G(i: u64) -> Key { Key::new('g', i) }
#[rustfmt::skip]
#[allow(non_snake_case)]
pub fn H(i: u64) -> Key { Key::new('h', i) }
#[rustfmt::skip]
#[allow(non_snake_case)]
pub fn I(i: u64) -> Key { Key::new('i', i) }
#[rustfmt::skip]
#[allow(non_snake_case)]
pub fn J(i: u64) -> Key { Key::new('j', i) }
#[rustfmt::skip]
#[allow(non_snake_case)]
pub fn K(i: u64) -> Key { Key::new('k', i) }
#[rustfmt::skip]
#[allow(non_snake_case)]
pub fn L(i: u64) -> Key { Key::new('l', i) }
#[rustfmt::skip]
#[allow(non_snake_case)]
pub fn M(i: u64) -> Key { Key::new('m', i) }
#[rustfmt::skip]
#[allow(non_snake_case)]
pub fn N(i: u64) -> Key { Key::new('n', i) }
#[rustfmt::skip]
#[allow(non_snake_case)]
pub fn O(i: u64) -> Key { Key::new('o', i) }
#[rustfmt::skip]
#[allow(non_snake_case)]
pub fn P(i: u64) -> Key { Key::new('p', i) }
#[rustfmt::skip]
#[allow(non_snake_case)]
pub fn Q(i: u64) -> Key { Key::new('q', i) }
#[rustfmt::skip]
#[allow(non_snake_case)]
pub fn R(i: u64) -> Key { Key::new('r', i) }
#[rustfmt::skip]
#[allow(non_snake_case)]
pub fn S(i: u64) -> Key { Key::new('s', i) }
#[rustfmt::skip]
#[allow(non_snake_case)]
pub fn T(i: u64) -> Key { Key::new('t', i) }
#[rustfmt::skip]
#[allow(non_snake_case)]
pub fn U(i: u64) -> Key { Key::new('u', i) }
#[rustfmt::skip]
#[allow(non_snake_case)]
pub fn V(i: u64) -> Key { Key::new('v', i) }
#[rustfmt::skip]
#[allow(non_snake_case)]
pub fn W(i: u64) -> Key { Key::new('w', i) }
#[rustfmt::skip]
#[allow(non_snake_case)]
pub fn X(i: u64) -> Key { Key::new('x', i) }
#[rustfmt::skip]
#[allow(non_snake_case)]
pub fn Y(i: u64) -> Key { Key::new('y', i) }
#[rustfmt::skip]
#[allow(non_snake_case)]
pub fn Z(i: u64) -> Key { Key::new('z', i) }
