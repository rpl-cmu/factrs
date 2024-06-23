// ------------------------- Import all variable types ------------------------- //
mod traits;
pub use traits::{LieGroup, Variable};

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
    use crate::{dtype, linalg::VectorX};
    use matrixcompare::assert_matrix_eq;

    use super::*;

    // Return a misc element for our tests
    fn element<T: Variable>(scale: dtype) -> T {
        let xi = VectorX::from_fn(T::DIM, |_, i| scale * ((i + 1) as dtype) / 10.0);
        T::exp(xi.as_view())
    }

    // TODO: Find a way to expose this better
    macro_rules! assert_variable_eq {
        ($x:expr, $y:expr) => {
            assert_matrix_eq!($x.ominus(&$y), VectorX::zeros($x.dim()));
        };
        ($x:expr, $y:expr, comp = exact) => {
            assert_matrix_eq!($x.ominus(&$y), VectorX::zeros($x.dim()), comp = exact);
        };
        ($x:expr, $y:expr, comp = abs, tol = $tol:expr) => {
            assert_matrix_eq!(
                $x.ominus(&$y),
                VectorX::zeros($x.dim()),
                comp = abs,
                tol = $tol
            );
        };
        ($x:expr, $y:expr, comp = ulp, tol = $tol:expr) => {
            assert_matrix_eq!(
                $x.ominus(&$y),
                VectorX::zeros($x.dim()),
                comp = ulp,
                tol = $tol
            );
        };
        ($x:expr, $y:expr, comp = float) => {
            assert_matrix_eq!($x.ominus(&$y), VectorX::zeros($x.dim()), comp = float);
        };
        ($x:expr, $y:expr, comp = float, $($key:ident = $val:expr),+) => {
            assert_matrix_eq!(
                $x.ominus(&$y),
                VectorX::zeros($x.dim()),
                comp = float,
                $($key:ident = $val:expr),+
            );
        };
    }
    // Try to test all the lie group rules
    // Closure should come by default (test manifold structure for SO(3) somehow?)
    // identity
    // inverse
    // associativity
    // exp/log are invertible near the origin

    macro_rules! variable_tests {
        ($($var:ident),*) => {
            use paste::paste;
            paste! {
                $(
                    #[test]
                    #[allow(non_snake_case)]
                    fn [<$var _identity>]() {
                        let var : $var = element(1.0);
                        let id = <$var as Variable>::identity();
                        assert_variable_eq!(var, var.compose(&id), comp = abs, tol = 1e-6);
                        assert_variable_eq!(var, id.compose(&var), comp = abs, tol = 1e-6);
                    }

                    #[test]
                    #[allow(non_snake_case)]
                    fn [<$var _inverse>]() {
                        let var : $var = element(1.0);
                        let inv = Variable::inverse(&var);
                        let id = <$var as Variable>::identity();
                        println!("{:?}", var);
                        println!("{:?}", inv);
                        assert_variable_eq!(var.compose(&inv), id, comp = abs, tol = 1e-6);
                        assert_variable_eq!(inv.compose(&var), id, comp = abs, tol = 1e-6);
                    }

                    #[test]
                    #[allow(non_snake_case)]
                    fn [<$var _associativity>]() {
                        let var1 : $var = element(1.0);
                        let var2 : $var = element(2.0);
                        let var3 : $var = element(3.0);
                        assert_variable_eq!(var1.compose(&var2).compose(&var3), var1.compose(&var2.compose(&var3)), comp = abs, tol = 1e-6);
                    }

                    #[test]
                    #[allow(non_snake_case)]
                    fn [<$var _exp_log>]() {
                        let var : $var = element(1.0);
                        let out = <$var as Variable>::exp(var.log().as_view());
                        assert_variable_eq!(var, out, comp = abs, tol = 1e-6);
                    }
                )*
            }
        };
    }

    variable_tests!(Vector1, Vector2, Vector3, Vector4, Vector5, Vector6, SO2, SE2, SO3, SE3);
}
