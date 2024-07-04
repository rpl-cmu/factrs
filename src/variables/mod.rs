// ------------------------- Import all variable types ------------------------- //
mod traits;
pub use traits::{MatrixLieGroup, Variable, VariableSafe};

mod so2;
pub use so2::SO2;

mod se2;
pub use se2::SE2;

mod so3;
pub use so3::SO3;

mod se3;
pub use se3::SE3;

mod vector;
pub use vector::{VectorVar1, VectorVar2, VectorVar3, VectorVar4, VectorVar5, VectorVar6};

mod macros;
