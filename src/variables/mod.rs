// ------------------------- Import all variable types ------------------------- //

mod symbol;
pub use symbol::*;

mod so3;
pub use so3::SO3;

mod se3;
pub use se3::SE3;

mod vector;
pub use crate::linalg::{
    Vector1, Vector10, Vector2, Vector3, Vector4, Vector5, Vector6, Vector7, Vector8, Vector9,
};

mod macros;
use crate::make_enum_variable;
make_enum_variable!(
    VariableEnum,
    SO3,
    SE3,
    Vector1,
    Vector2,
    Vector3,
    Vector4,
    Vector5,
    Vector6,
    Vector7,
    Vector8,
    Vector9,
    Vector10
);
