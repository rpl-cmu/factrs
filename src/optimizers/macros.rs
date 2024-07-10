#[macro_export]
macro_rules! test_optimizer {
    ($optimizer:ident $(<$($gen:ident),* >)?) => {
        #[test]
        fn priorvector3() {
            $crate::optimizers::test::optimize_prior::<$optimizer$(< $($gen),* >)?, $crate::variables::VectorVar3, 3>()
        }

        #[test]
        fn priorso3() {
            $crate::optimizers::test::optimize_prior::<$optimizer$(< $($gen),* >)?, $crate::variables::SO3, 3>();
        }

        #[test]
        fn priorse3() {
            $crate::optimizers::test::optimize_prior::<$optimizer$(< $($gen),* >)?, $crate::variables::SE3, 6>();
        }

        #[test]
        fn betweenvector3() {
            $crate::optimizers::test::optimize_between::<$optimizer$(< $($gen),* >)?, $crate::variables::VectorVar3, 3, 6>();
        }

        #[test]
        fn betweenso3() {
            $crate::optimizers::test::optimize_between::<$optimizer$(< $($gen),* >)?, $crate::variables::SO3, 3, 6>();
        }

        #[test]
        fn betweense3() {
            $crate::optimizers::test::optimize_between::<$optimizer$(< $($gen),* >)?, $crate::variables::SE3, 6, 12>();
        }
    };
}
