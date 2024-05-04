use crate::variables::{
    Variable, Vector1, Vector10, Vector2, Vector3, Vector4, Vector5, Vector6, Vector7, Vector8,
    Vector9, VectorD, SO3,
};
use std::convert::{From, TryFrom};

//
/*
TODO: I think eventually, this could all be macro-generated.

enum-dispatch is really close, but doesn't handle the const val, static methods, or returning Self.

A subtrait *almost* works, but it enum-dispatch doesn't work with supertraits

It would also be awesome if the macro could add any types to the enum, but I'm not sure that's possible
*/
pub enum VariableEnum {
    SO3(SO3),
    Vector1(Vector1),
    Vector2(Vector2),
    Vector3(Vector3),
    Vector4(Vector4),
    Vector5(Vector5),
    Vector6(Vector6),
    Vector7(Vector7),
    Vector8(Vector8),
    Vector9(Vector9),
    Vector10(Vector10),
}

impl VariableEnum {
    pub fn dim(&self) -> usize {
        match self {
            VariableEnum::SO3(_) => SO3::DIM,
            VariableEnum::Vector1(_) => Vector1::DIM,
            VariableEnum::Vector2(_) => Vector2::DIM,
            VariableEnum::Vector3(_) => Vector3::DIM,
            VariableEnum::Vector4(_) => Vector4::DIM,
            VariableEnum::Vector5(_) => Vector5::DIM,
            VariableEnum::Vector6(_) => Vector6::DIM,
            VariableEnum::Vector7(_) => Vector7::DIM,
            VariableEnum::Vector8(_) => Vector8::DIM,
            VariableEnum::Vector9(_) => Vector9::DIM,
            VariableEnum::Vector10(_) => Vector10::DIM,
        }
    }

    pub fn identity(&self) -> Self {
        match self {
            VariableEnum::SO3(_) => VariableEnum::SO3(SO3::identity()),
            VariableEnum::Vector1(_) => VariableEnum::Vector1(Vector1::identity()),
            VariableEnum::Vector2(_) => VariableEnum::Vector2(Vector2::identity()),
            VariableEnum::Vector3(_) => VariableEnum::Vector3(Vector3::identity()),
            VariableEnum::Vector4(_) => VariableEnum::Vector4(Vector4::identity()),
            VariableEnum::Vector5(_) => VariableEnum::Vector5(Vector5::identity()),
            VariableEnum::Vector6(_) => VariableEnum::Vector6(Vector6::identity()),
            VariableEnum::Vector7(_) => VariableEnum::Vector7(Vector7::identity()),
            VariableEnum::Vector8(_) => VariableEnum::Vector8(Vector8::identity()),
            VariableEnum::Vector9(_) => VariableEnum::Vector9(Vector9::identity()),
            VariableEnum::Vector10(_) => VariableEnum::Vector10(Vector10::identity()),
        }
    }

    pub fn inverse(&self) -> Self {
        match self {
            VariableEnum::SO3(so3) => VariableEnum::SO3(so3.inverse()),
            VariableEnum::Vector1(v) => VariableEnum::Vector1(v.inverse()),
            VariableEnum::Vector2(v) => VariableEnum::Vector2(v.inverse()),
            VariableEnum::Vector3(v) => VariableEnum::Vector3(v.inverse()),
            VariableEnum::Vector4(v) => VariableEnum::Vector4(v.inverse()),
            VariableEnum::Vector5(v) => VariableEnum::Vector5(v.inverse()),
            VariableEnum::Vector6(v) => VariableEnum::Vector6(v.inverse()),
            VariableEnum::Vector7(v) => VariableEnum::Vector7(v.inverse()),
            VariableEnum::Vector8(v) => VariableEnum::Vector8(v.inverse()),
            VariableEnum::Vector9(v) => VariableEnum::Vector9(v.inverse()),
            VariableEnum::Vector10(v) => VariableEnum::Vector10(v.inverse()),
        }
    }

    pub fn oplus(&self, delta: &VectorD) -> Self {
        match self {
            VariableEnum::SO3(so3) => VariableEnum::SO3(so3.oplus(delta)),
            VariableEnum::Vector1(v) => VariableEnum::Vector1(v.oplus(delta)),
            VariableEnum::Vector2(v) => VariableEnum::Vector2(v.oplus(delta)),
            VariableEnum::Vector3(v) => VariableEnum::Vector3(v.oplus(delta)),
            VariableEnum::Vector4(v) => VariableEnum::Vector4(v.oplus(delta)),
            VariableEnum::Vector5(v) => VariableEnum::Vector5(v.oplus(delta)),
            VariableEnum::Vector6(v) => VariableEnum::Vector6(v.oplus(delta)),
            VariableEnum::Vector7(v) => VariableEnum::Vector7(v.oplus(delta)),
            VariableEnum::Vector8(v) => VariableEnum::Vector8(v.oplus(delta)),
            VariableEnum::Vector9(v) => VariableEnum::Vector9(v.oplus(delta)),
            VariableEnum::Vector10(v) => VariableEnum::Vector10(v.oplus(delta)),
        }
    }

    pub fn ominus(&self, other: &Self) -> VectorD {
        // TODO: Should this return a result instead? In case the types don't match? Currently it just panics
        match self {
            VariableEnum::SO3(so3) => so3.ominus(other.try_into().unwrap()),
            VariableEnum::Vector1(v) => v.ominus(other.try_into().unwrap()),
            VariableEnum::Vector2(v) => v.ominus(other.try_into().unwrap()),
            VariableEnum::Vector3(v) => v.ominus(other.try_into().unwrap()),
            VariableEnum::Vector4(v) => v.ominus(other.try_into().unwrap()),
            VariableEnum::Vector5(v) => v.ominus(other.try_into().unwrap()),
            VariableEnum::Vector6(v) => v.ominus(other.try_into().unwrap()),
            VariableEnum::Vector7(v) => v.ominus(other.try_into().unwrap()),
            VariableEnum::Vector8(v) => v.ominus(other.try_into().unwrap()),
            VariableEnum::Vector9(v) => v.ominus(other.try_into().unwrap()),
            VariableEnum::Vector10(v) => v.ominus(other.try_into().unwrap()),
        }
    }
}

// ------------------------- Converting Variables -> VariableEnum ------------------------- //
macro_rules! enum_from_var {
    ( $e:path, $x:ty) => {
        impl From<$x> for VariableEnum {
            fn from(a: $x) -> Self {
                $e(a)
            }
        }
    };
}

enum_from_var!(VariableEnum::SO3, SO3);
enum_from_var!(VariableEnum::Vector1, Vector1);
enum_from_var!(VariableEnum::Vector2, Vector2);
enum_from_var!(VariableEnum::Vector3, Vector3);
enum_from_var!(VariableEnum::Vector4, Vector4);
enum_from_var!(VariableEnum::Vector5, Vector5);
enum_from_var!(VariableEnum::Vector6, Vector6);
enum_from_var!(VariableEnum::Vector7, Vector7);
enum_from_var!(VariableEnum::Vector8, Vector8);
enum_from_var!(VariableEnum::Vector9, Vector9);
enum_from_var!(VariableEnum::Vector10, Vector10);

// TODO: Get ref working in the other direction?

// ------------------------- Convert VariableEnum -> Variable ------------------------- //
macro_rules! var_tryfrom_enum {
    ( $x:ty, $e: path ) => {
        impl TryFrom<VariableEnum> for $x {
            type Error = &'static str;

            fn try_from(value: VariableEnum) -> Result<Self, Self::Error> {
                match value {
                    $e(v) => Ok(v),
                    _ => Err("fix me later"),
                }
            }
        }
    };
}

var_tryfrom_enum!(SO3, VariableEnum::SO3);
var_tryfrom_enum!(Vector1, VariableEnum::Vector1);
var_tryfrom_enum!(Vector2, VariableEnum::Vector2);
var_tryfrom_enum!(Vector3, VariableEnum::Vector3);
var_tryfrom_enum!(Vector4, VariableEnum::Vector4);
var_tryfrom_enum!(Vector5, VariableEnum::Vector5);
var_tryfrom_enum!(Vector6, VariableEnum::Vector6);
var_tryfrom_enum!(Vector7, VariableEnum::Vector7);
var_tryfrom_enum!(Vector8, VariableEnum::Vector8);
var_tryfrom_enum!(Vector9, VariableEnum::Vector9);
var_tryfrom_enum!(Vector10, VariableEnum::Vector10);

macro_rules! var_tryfrom_enum_ref {
    ( $x:ty, $e: path ) => {
        impl<'a> TryFrom<&'a VariableEnum> for &'a $x {
            type Error = &'static str;

            fn try_from(value: &'a VariableEnum) -> Result<Self, Self::Error> {
                match value {
                    $e(v) => Ok(v),
                    _ => Err("fix me later"),
                }
            }
        }
    };
}

var_tryfrom_enum_ref!(SO3, VariableEnum::SO3);
var_tryfrom_enum_ref!(Vector1, VariableEnum::Vector1);
var_tryfrom_enum_ref!(Vector2, VariableEnum::Vector2);
var_tryfrom_enum_ref!(Vector3, VariableEnum::Vector3);
var_tryfrom_enum_ref!(Vector4, VariableEnum::Vector4);
var_tryfrom_enum_ref!(Vector5, VariableEnum::Vector5);
var_tryfrom_enum_ref!(Vector6, VariableEnum::Vector6);
var_tryfrom_enum_ref!(Vector7, VariableEnum::Vector7);
var_tryfrom_enum_ref!(Vector8, VariableEnum::Vector8);
var_tryfrom_enum_ref!(Vector9, VariableEnum::Vector9);
var_tryfrom_enum_ref!(Vector10, VariableEnum::Vector10);
