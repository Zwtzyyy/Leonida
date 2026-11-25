fn main() {
    println!("cargo:rustc-link-arg-cdylib=/DELAYLOAD:vicious-probe-delayed.dll");
    println!("cargo:rustc-link-lib=delayimp");
}
