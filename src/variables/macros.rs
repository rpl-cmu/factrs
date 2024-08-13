/// Variable wrapper around [assert_matrix_eq](matrixcompare::assert_matrix_eq)
///
/// This compares two variables using
/// [ominus](crate::variables::Variable::ominus)
#[macro_export]
macro_rules! assert_variable_eq {
    ($x:expr, $y:expr) => {
        matrixcompare::assert_matrix_eq!($x.ominus(&$y), VectorX::zeros($x.dim()));
    };
    ($x:expr, $y:expr, comp = exact) => {
        matrixcompare::assert_matrix_eq!($x.ominus(&$y), VectorX::zeros($x.dim()), comp = exact);
    };
    ($x:expr, $y:expr, comp = abs, tol = $tol:expr) => {
        matrixcompare::assert_matrix_eq!(
            $x.ominus(&$y),
            VectorX::zeros($crate::variables::traits::Variable::dim(&$x)),
            comp = abs,
            tol = $tol
        );
    };
    ($x:expr, $y:expr, comp = ulp, tol = $tol:expr) => {
        matrixcompare::assert_matrix_eq!(
            $x.ominus(&$y),
            VectorX::zeros($x.dim()),
            comp = ulp,
            tol = $tol
        );
    };
    ($x:expr, $y:expr, comp = float) => {
        matrixcompare::assert_matrix_eq!($x.ominus(&$y), VectorX::zeros($x.dim()), comp = float);
    };
    ($x:expr, $y:expr, comp = float, $($key:ident = $val:expr),+) => {
        matrixcompare::assert_matrix_eq!(
            $x.ominus(&$y),
            VectorX::zeros($x.dim()),
            comp = float,
            $($key:ident = $val:expr),+
        );
    };
}

/// Test (most of) the lie group rules
///
/// Specifically this tests:
/// - identity
/// - inverse
/// - associativity
/// - exp/log are invertible near the origin
#[macro_export]
macro_rules! test_variable {
    ($var:ident) => {
        // Return a misc element for our tests
        fn element<T: Variable>(scale: $crate::dtype) -> T {
            let xi = VectorX::from_fn(T::DIM, |_, i| scale * ((i + 1) as $crate::dtype) / 10.0);
            T::exp(xi.as_view())
        }

        #[test]
        #[allow(non_snake_case)]
        fn identity() {
            let var: $var = element(1.0);
            let id = <$var as Variable>::identity();
            $crate::assert_variable_eq!(var, var.compose(&id), comp = abs, tol = 1e-6);
            $crate::assert_variable_eq!(var, id.compose(&var), comp = abs, tol = 1e-6);
        }

        #[test]
        #[allow(non_snake_case)]
        fn inverse() {
            let var: $var = element(1.0);
            let inv = Variable::inverse(&var);
            let id = <$var as Variable>::identity();
            println!("{:?}", var);
            println!("{:?}", inv);
            $crate::assert_variable_eq!(var.compose(&inv), id, comp = abs, tol = 1e-6);
            $crate::assert_variable_eq!(inv.compose(&var), id, comp = abs, tol = 1e-6);
        }

        #[test]
        #[allow(non_snake_case)]
        fn associativity() {
            let var1: $var = element(1.0);
            let var2: $var = element(2.0);
            let var3: $var = element(3.0);
            $crate::assert_variable_eq!(
                var1.compose(&var2).compose(&var3),
                var1.compose(&var2.compose(&var3)),
                comp = abs,
                tol = 1e-6
            );
        }

        #[test]
        #[allow(non_snake_case)]
        fn exp_log() {
            let var: $var = element(1.0);
            let out = <$var as Variable>::exp(var.log().as_view());
            $crate::assert_variable_eq!(var, out, comp = abs, tol = 1e-6);
        }
    };
}

/// Test (most of) the matrix lie group rules
///
/// Specifcally test
/// - to/form matrix
/// - hat/vee
/// - jacobian of rotation function with hat_swap
#[macro_export]
macro_rules! test_lie {
    ($var:ident) => {
        fn tangent<T: Variable>(scale: dtype) -> $crate::linalg::VectorX {
            VectorX::from_fn(T::DIM, |_, i| scale * ((i + 1) as dtype) / 10.0)
        }

        #[test]
        fn matrix() {
            let tan = tangent::<$var>(1.0);
            let var = $var::exp(tan.as_view());
            let mat = var.to_matrix();
            let var_after = $var::from_matrix(mat.as_view());
            $crate::assert_variable_eq!(var, var_after, comp = abs, tol = 1e-6);
        }

        #[test]
        fn hat_vee() {
            let tan = tangent::<$var>(1.0);
            let mat = $var::hat(tan.as_view());
            let tan_after = $var::vee(mat.as_view());
            matrixcompare::assert_matrix_eq!(tan, tan_after);
        }

        // TODO: Someway to test rotate & adjoint functions?

        #[cfg(not(feature = "left"))]
        use $crate::linalg::{Diff, Dim};

        #[cfg(not(feature = "left"))]
        #[test]
        fn jacobian() {
            let vec_len =
                <$var as $crate::variables::MatrixLieGroup>::VectorDim::try_to_usize().unwrap();
            let v = VectorX::from_fn(vec_len, |i, _| (i + 1) as $crate::dtype);

            // Function that simply rotates a vector
            fn rotate<D: $crate::linalg::Numeric>(r: $var<D>) -> $crate::linalg::VectorX<D> {
                let vec_len =
                    <$var as $crate::variables::MatrixLieGroup>::VectorDim::try_to_usize().unwrap();
                let v = VectorX::from_fn(vec_len, |i, _| D::from((i + 1) as $crate::dtype));
                let rotated = r.apply(v.as_view());
                VectorX::from_fn(vec_len, |i, _| rotated[i].clone())
            }

            let t = $var::exp(tangent::<$var>(1.0).as_view());
            let $crate::linalg::DiffResult {
                value: _x,
                diff: dx,
            } = $crate::linalg::ForwardProp::<<$var as Variable>::Dim>::jacobian_1(rotate, &t);

            let size =
                <$var as $crate::variables::MatrixLieGroup>::VectorDim::try_to_usize().unwrap();
            let dx_exp = t.to_matrix().view((0, 0), (size, size)) * $var::hat_swap(v.as_view());

            println!("Expected: {}", dx_exp);
            println!("Actual: {}", dx);

            matrixcompare::assert_matrix_eq!(dx, dx_exp, comp = abs, tol = 1e-6);
        }
    };
}

/// Register a type as a [variable](crate::variables) for serialization.
#[macro_export]
macro_rules! tag_variable {
    ($($ty:ty),* $(,)?) => {$(
        $crate::register_typetag!($crate::variables::VariableSafe, $ty);
    )*};
}
