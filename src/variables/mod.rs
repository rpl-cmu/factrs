//! Variables to optimize
//!
//! This module contains the definition of the variables that can be optimized
//! using the optimization algorithms. We model each variable as a Lie group
//! $\mathcal{G}$, even if it is trivially so. Because of this, each type $X \in
//! \mathcal{G}$ must satisfy the following properties,
//!
//! Identity
//! $$
//! \exists I \ni X \cdot I = X = I \cdot X
//! $$
//! Inverse
//! $$
//! \exists X^{-1} \ni X \cdot X^{-1} = I = X^{-1} \cdot X
//! $$
//! Closed under group composition
//! $$
//! \exists X_1, X_2 \ni X_1 \cdot X_2 = X_3
//! $$
//! Associativity
//! $$
//! \forall X_1, X_2, X_3 \ni (X_1 \cdot X_2) \cdot X_3 = X_1 \cdot (X_2 \cdot
//! X_3) $$
//! Exponential and Logarithm maps. While not part of the group definition, we
//! require each type to implement these for optimization.
//! $$
//! \xi \in \mathfrak{g} \implies \exp(\xi) \in \mathcal{G}
//! $$
//! $$
//! X \in \mathcal{G} \implies \log(X) \in \mathfrak{g}
//! $$
//! Finally, for optimization purposes, we adopt $\oplus$ and $\ominus$
//! operators as defined in "Micro Lie Theory" by Joan Solà. By default this
//! results in,
//!
//! $$
//! x \oplus \xi = x \cdot \exp(\xi) \\\\
//! x \ominus y = \log(y^{-1} \cdot x)
//! $$
//! If the "left" feature is enabled, instead this turns to
//! $$
//! x \oplus \xi = \exp(\xi) \cdot x \\\\
//! x \ominus y = \log(x \cdot y^{-1})
//! $$
//!
//! All these properties are encapsulated in the [Variable] trait. Additionally,
//! we parametrized each variable over its datatype to allow for dual numbers to
//! be propagated through residuals for jacobian computation.
//!
//! If you want to implement a custom variable, you'll need to implement
//! [Variable] and call the [tag_variable](crate::tag_variable) macro if using
//! serde. We also recommend using the [test_variable](crate::test_variable)
//! macro to ensure these properties are satisfied.
mod traits;
pub use traits::{MatrixLieGroup, Variable, VariableSafe, VariableUmbrella};

mod so2;
pub use so2::SO2;

mod se2;
pub use se2::SE2;

mod so3;
pub use so3::SO3;

mod se3;
pub use se3::SE3;

mod vector;
pub use vector::{
    VectorVar,
    VectorVar1,
    VectorVar2,
    VectorVar3,
    VectorVar4,
    VectorVar5,
    VectorVar6,
};

mod imu_bias;
pub use imu_bias::ImuBias;

mod macros;
