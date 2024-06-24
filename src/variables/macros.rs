#[macro_export]
macro_rules! make_enum_variable {
    ( $name:ident$(,)? ) => {};

    ( $name:ident, $( $x:ident),* $(,)?) => {
        #[derive(Clone, derive_more::From, derive_more::TryInto)]
        #[try_into(owned, ref, ref_mut)]
        pub enum $name<D: $crate::linalg::DualNum = $crate::dtype> {
            $(
                $x($x<D>),
            )*
        }

        impl <D: $crate::linalg::DualNum> std::fmt::Display for $name<D> {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                match self {
                    $(
                        $name::$x(x) => write!(f, "{:?}", x),
                    )*
                }
            }
        }

        impl <D: $crate::linalg::DualNum> std::fmt::Debug for $name<D> {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                std::fmt::Display::fmt(self, f)
            }
        }

        // Implement the trait for each enum
        impl<D: $crate::linalg::DualNum> $crate::variables::Variable<D> for $name<D> {
            const DIM: usize = 0;
            type Dual = $name<$crate::linalg::DualVec>;

            fn dim(&self) -> usize {
                match self {
                    $(
                        $name::$x(x) => x.dim(),
                    )*
                }
            }

            fn identity() -> Self {
                panic!("Cannot call static identity on enum")
            }

            fn inverse(&self) -> Self {
                match self {
                    $(
                        $name::$x(x) => $name::$x(x.inverse()),
                    )*
                }
            }

            fn compose(&self, other: &Self) -> Self {
                match (self, other) {
                    $(
                        ($name::$x(x), $name::$x(y)) => $name::$x(x.compose(y)),
                    )*
                    _ => panic!("Cannot compose different types"),
                }
            }

            fn exp(_delta: $crate::linalg::VectorViewX<D>) -> Self {
                panic!("Cannot call static exp on enum")
            }

            fn log(&self) -> $crate::linalg::VectorX<D> {
                match self {
                    $(
                        $name::$x(x) => x.log(),
                    )*
                }
            }

            fn dual_self(&self) -> Self::Dual {
                match self {
                    $(
                        $name::$x(x) => $name::<$crate::linalg::DualVec>::$x(x.dual_self()),
                    )*
                }
            }

            // Overrides for various enum helpers
            // For some of these, they work fine without this, but
            // it's preferable we delegate to underlying implementation to reduce # of matches we have to do
            fn identity_enum(&self) -> Self {
                match self {
                    $(
                        $name::$x(_) => $name::$x($x::identity()),
                    )*
                }
            }

            fn oplus(&self, delta: $crate::linalg::VectorViewX<D>) -> Self {
                match self {
                    $(
                        $name::$x(x) => $name::$x(x.oplus(delta)),
                    )*
                }
            }

            fn ominus(&self, other: &Self) -> $crate::linalg::VectorX<D> {
                match (self, other) {
                    $(
                        ($name::$x(x), $name::$x(y)) => x.ominus(y),
                    )*
                    _ => panic!("Cannot ominus different types"),
                }
            }

            fn dual_tangent(&self, idx: usize, total: usize) -> $crate::linalg::VectorX<$crate::linalg::DualVec> {
                match self {
                    $(
                        $name::$x(x) => x.dual_tangent(idx, total),
                    )*
                }
            }

            fn dual(&self, idx: usize, total: usize) -> Self::Dual {
                match self {
                    $(
                        $name::$x(x) => $name::$x(x.dual(idx, total)),
                    )*
                }
            }
        }

    };
}

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
            VectorX::zeros($x.dim()),
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

// Try to test all the lie group rules
// Closure should come by default (test manifold structure for SO(3) somehow?)
// identity
// inverse
// associativity
// exp/log are invertible near the origin
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

// Check Lie Algebra rules
#[macro_export]
macro_rules! test_lie {
    ($var:ident) => {
        use $crate::linalg::{Diff, Dim};

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
        #[test]
        fn jacobian() {
            let vec_len =
                <$var as $crate::variables::MatrixLieGroup>::VectorDim::try_to_usize().unwrap();
            let v = VectorX::from_fn(vec_len, |i, _| (i + 1) as $crate::dtype);

            // Function that simply rotates a vector
            fn rotate<D: $crate::linalg::DualNum>(r: $var<D>) -> $crate::linalg::VectorX<D> {
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
            } = $crate::linalg::ForwardProp::jacobian_1(rotate, &t);

            let size =
                <$var as $crate::variables::MatrixLieGroup>::VectorDim::try_to_usize().unwrap();
            let dx_exp = t.to_matrix().view((0, 0), (size, size)) * $var::hat_swap(v.as_view());

            println!("Expected: {}", dx_exp);
            println!("Actual: {}", dx);

            matrixcompare::assert_matrix_eq!(dx, dx_exp, comp = abs, tol = 1e-6);
        }
    };
}
