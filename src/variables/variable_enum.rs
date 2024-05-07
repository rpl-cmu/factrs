use crate::variables::{
    Variable, Vector1, Vector10, Vector2, Vector3, Vector4, Vector5, Vector6, Vector7, Vector8,
    Vector9, VectorD, SE3, SO3,
};
use std::convert::{Into, TryFrom};
use std::fmt;

//
/*
TODO: I think eventually, this could all be macro-generated.

enum-dispatch is really close, but doesn't handle the const val, static methods, or returning Self.

A subtrait *almost* works, but it enum-dispatch doesn't work with supertraits

It would also be awesome if the macro could add any types to the enum, but I'm not sure that's possible
*/
pub trait VariableEnumDispatch: Into<VariableEnum> + TryFrom<VariableEnum> {}

#[derive(Clone)]
pub enum VariableEnum {
    SO3(SO3),
    SE3(SE3),
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

impl fmt::Display for VariableEnum {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            VariableEnum::SO3(a) => write!(f, "{}", a),
            VariableEnum::SE3(a) => write!(f, "{}", a),
            VariableEnum::Vector1(a) => write!(f, "Vector1{:?}", a),
            VariableEnum::Vector2(a) => write!(f, "Vector2{:?}", a),
            VariableEnum::Vector3(a) => write!(f, "Vector3{:?}", a),
            VariableEnum::Vector4(a) => write!(f, "Vector4{:?}", a),
            VariableEnum::Vector5(a) => write!(f, "Vector5{:?}", a),
            VariableEnum::Vector6(a) => write!(f, "Vector6{:?}", a),
            VariableEnum::Vector7(a) => write!(f, "Vector7{:?}", a),
            VariableEnum::Vector8(a) => write!(f, "Vector8{:?}", a),
            VariableEnum::Vector9(a) => write!(f, "Vector9{:?}", a),
            VariableEnum::Vector10(a) => write!(f, "Vector10{:?}", a),
        }
    }
}

impl VariableEnumDispatch for VariableEnum {}

impl VariableEnum {
    pub fn dim(&self) -> usize {
        match self {
            VariableEnum::SO3(_) => SO3::DIM,
            VariableEnum::SE3(_) => SE3::DIM,
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
            VariableEnum::SE3(_) => VariableEnum::SE3(SE3::identity()),
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
            VariableEnum::SE3(se3) => VariableEnum::SE3(se3.inverse()),
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
            VariableEnum::SE3(se3) => VariableEnum::SE3(se3.oplus(delta)),
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
            VariableEnum::SE3(se3) => se3.ominus(other.try_into().unwrap()),
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
// TODO: Define custom error type for this?
macro_rules! enum_trait_impl {
    ( $e:path, $x:ty) => {
        impl VariableEnumDispatch for $x {}

        impl TryFrom<VariableEnum> for $x {
            type Error = String;

            fn try_from(value: VariableEnum) -> Result<Self, Self::Error> {
                match value {
                    $e(v) => Ok(v),
                    _ => Err(format!("{} can't be turned into {}", value, stringify!($x))),
                }
            }
        }

        impl<'a> TryFrom<&'a VariableEnum> for &'a $x {
            type Error = String;

            fn try_from(value: &'a VariableEnum) -> Result<Self, Self::Error> {
                match value {
                    $e(v) => Ok(v),
                    _ => Err(format!("{} can't be turned into {}", value, stringify!($x))),
                }
            }
        }

        impl Into<VariableEnum> for $x {
            fn into(self) -> VariableEnum {
                $e(self)
            }
        }
    };
}

enum_trait_impl!(VariableEnum::SO3, SO3);
enum_trait_impl!(VariableEnum::SE3, SE3);
enum_trait_impl!(VariableEnum::Vector1, Vector1);
enum_trait_impl!(VariableEnum::Vector2, Vector2);
enum_trait_impl!(VariableEnum::Vector3, Vector3);
enum_trait_impl!(VariableEnum::Vector4, Vector4);
enum_trait_impl!(VariableEnum::Vector5, Vector5);
enum_trait_impl!(VariableEnum::Vector6, Vector6);
enum_trait_impl!(VariableEnum::Vector7, Vector7);
enum_trait_impl!(VariableEnum::Vector8, Vector8);
enum_trait_impl!(VariableEnum::Vector9, Vector9);
enum_trait_impl!(VariableEnum::Vector10, Vector10);
