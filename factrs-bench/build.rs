#[cfg(feature = "cpp")]
fn main() {
    // Handle dependencies via CMake fetch
    // Turn off dependency warnings cuz we don't care
    let dst = cmake::Config::new(".").cxxflag("-w").build();

    println!("cargo:rustc-link-search=native={}", dst.display());
    println!("cargo:rustc-link-lib=static=gtsam");

    // Build our simple cpp scripts
    cxx_build::bridge("src/lib.rs")
        .file("src/gtsam.cpp")
        .std("c++17")
        .include(format!("{}/include", dst.display()))
        .include(format!("{}/include/gtsam/3rdparty/Eigen", dst.display()))
        .flag("-Wno-deprecated-copy")
        .flag("-Wno-unused-parameter")
        .flag("-Wno-deprecated-declarations")
        .compile("cpp_benches");

    // Tell Cargo that if the given file changes, to rerun this build script.
    println!("cargo:rerun-if-changed=CMakeLists.txt");
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/gtsam.cpp");
    println!("cargo:rerun-if-changed=include/gtsam.h");
}

#[cfg(not(feature = "cpp"))]
fn main() {}
