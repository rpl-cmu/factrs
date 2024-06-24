// ------------------------- Import all variable types ------------------------- //
mod traits;
pub use traits::{MatrixLieGroup, Variable};

mod so2;
pub use so2::SO2;

mod se2;
pub use se2::SE2;

mod so3;
pub use so3::SO3;

mod se3;
pub use se3::SE3;

mod vector;
pub use crate::linalg::{Vector1, Vector2, Vector3, Vector4, Vector5, Vector6};

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
    Vector6
);

// ------------------------- Assert Variables are Equal ------------------------- //

#[cfg(test)]
mod test {
    
    

    

    // TODO: Find a way to expose this better
    // Try to test all the lie group rules
    // Closure should come by default (test manifold structure for SO(3) somehow?)
    // identity
    // inverse
    // associativity
    // exp/log are invertible near the origin

    // variable_tests!(Vector1, Vector2, Vector3, Vector4, Vector5, Vector6, SO2, SE2, SO3, SE3);
}
