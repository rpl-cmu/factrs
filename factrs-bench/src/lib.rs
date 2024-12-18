// pub mod tiny_solver;
// mod sophus;

#[cfg(feature = "cpp")]
#[cxx::bridge]
pub mod gtsam {
    unsafe extern "C++" {
        include!("factrs-bench/include/gtsam.h");

        fn gtsam_hello();
    }
}
