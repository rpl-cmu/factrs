/// Test for optimizers
///
/// This macro generates a handful of sanity tests for an optimizer. It tests
/// - Prior optimization for VectorVar3, SO3, and SE3
/// - Between optimization for VectorVar3, SO3, and SE3
#[macro_export]
macro_rules! test_optimizer {
    ($optimizer:ty) => {
        #[test]
        fn priorvector3() {
            $crate::optimizers::test::optimize_prior::<$optimizer, 3, $crate::variables::VectorVar3>()
        }

        #[test]
        fn priorso3() {
            $crate::optimizers::test::optimize_prior::<$optimizer, 3, $crate::variables::SO3>();
        }

        #[test]
        fn priorse3() {
            $crate::optimizers::test::optimize_prior::<$optimizer, 6, $crate::variables::SE3>();
        }

        #[test]
        fn betweenvector3() {
            $crate::optimizers::test::optimize_between::<$optimizer, 3, 6, $crate::variables::VectorVar3>();
        }

        #[test]
        fn betweenso3() {
            $crate::optimizers::test::optimize_between::<$optimizer, 3, 6, $crate::variables::SO3>();
        }

        #[test]
        fn betweense3() {
            $crate::optimizers::test::optimize_between::<$optimizer, 6, 12, $crate::variables::SE3>();
        }
    };
}
