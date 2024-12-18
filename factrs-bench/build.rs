fn main() {
    #[cfg(feature = "cpp")]
    let dst = cmake::build(".");

    #[cfg(feature = "cpp")]
    cxx_build::bridge("src/lib.rs")
        .file("src/gtsam.cpp")
        .std("c++17")
        .include(format!("{}/include", dst.display()))
        .include(format!("{}/include/gtsam/3rdparty/Eigen", dst.display()))
        .compile("cpp_benches");

    // Tell Cargo that if the given file changes, to rerun this build script.
    println!("cargo:rerun-if-changed=CMakeLists.txt");
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/gtsam.cpp");
    println!("cargo:rerun-if-changed=include/gtsam.h");

    #[cfg(feature = "cpp")]
    println!("cargo:rustc-link-search=native={}", dst.display());
    #[cfg(feature = "cpp")]
    println!("cargo:rustc-link-lib=static=gtsam");
}
