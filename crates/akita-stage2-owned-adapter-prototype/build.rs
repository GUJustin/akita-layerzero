use std::{env, fs, path::PathBuf, process::Command};

fn run(command: &mut Command) {
    let status = command.status().expect("launch native Stage2 compiler");
    assert!(status.success(), "native Stage2 build failed: {command:?}");
}

fn main() {
    println!("cargo:rerun-if-env-changed=AKITA_STAGE2_NATIVE_LIB_DIR");
    println!("cargo:rerun-if-env-changed=SDKROOT");
    if env::var("TARGET").unwrap() != "aarch64-apple-darwin" {
        assert!(
            env::var_os("AKITA_STAGE2_NATIVE_LIB_DIR").is_none(),
            "native Stage2 target"
        );
        return;
    }
    let dir = if let Some(dir) = env::var_os("AKITA_STAGE2_NATIVE_LIB_DIR") {
        let dir = PathBuf::from(dir);
        assert!(dir.is_absolute(), "native library path must be absolute");
        dir
    } else {
        let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
        let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
        for file in [
            "native/owner.mm",
            "native/field.h",
            "native/stage2.metal",
            "include/stage2.h",
        ] {
            println!("cargo:rerun-if-changed={}", root.join(file).display());
        }
        let shader = fs::read_to_string(root.join("native/field.h")).unwrap()
            + &fs::read_to_string(root.join("native/stage2.metal")).unwrap();
        assert!(!shader.contains(")OWNED_STAGE2\""));
        fs::write(
            out.join("shader_source.h"),
            format!("static const char shader_source[]=R\"OWNED_STAGE2({shader})OWNED_STAGE2\";\n"),
        )
        .unwrap();
        let mut compile = Command::new("/usr/bin/clang++");
        compile.args(["-std=c++17", "-O2", "-fobjc-arc", "-fexceptions"]);
        if let Some(sdk) = env::var_os("SDKROOT") {
            compile.arg("-isysroot").arg(sdk);
        }
        compile
            .arg("-I")
            .arg(&out)
            .arg("-c")
            .arg(root.join("native/owner.mm"))
            .arg("-o")
            .arg(out.join("owner.o"));
        run(&mut compile);
        run(Command::new("/usr/bin/ar")
            .arg("rcs")
            .arg(out.join("libakita_stage2_owned.a"))
            .arg(out.join("owner.o")));
        out
    };
    let archive = dir.join("libakita_stage2_owned.a");
    assert!(archive.is_file(), "native archive must exist");
    println!("cargo:rerun-if-changed={}", archive.display());
    println!("cargo:rustc-link-search=native={}", dir.display());
    println!("cargo:rustc-link-lib=static=akita_stage2_owned");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=c++");
}
