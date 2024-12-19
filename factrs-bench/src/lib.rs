pub mod tiny_solver;
// mod sophus;
#[cfg(feature = "cpp")]
#[cxx::bridge]
pub mod gtsam {
    unsafe extern "C++" {
        include!("factrs-bench/include/gtsam.h");

        type GraphValues;

        fn load_g2o(file: &CxxString, is3D: bool) -> UniquePtr<GraphValues>;

        fn run(gv: &GraphValues);

        fn hello();
    }
}
