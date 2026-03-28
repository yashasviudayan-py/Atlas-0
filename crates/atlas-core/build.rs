//! Build script for atlas-core.
//!
//! Compiles `proto/atlas.proto` into Rust code using `prost-build` and the
//! pure-Rust `protox` compiler (no system `protoc` required).  The generated
//! file is written to `$OUT_DIR/atlas.rs` and included by `src/lib.rs`.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")?;

    // Navigate from crates/atlas-core/ → crates/ → workspace root.
    let workspace_root = std::path::Path::new(&manifest_dir)
        .parent() // crates/
        .expect("crate must be nested in crates/")
        .parent() // workspace root
        .expect("crates/ must be inside workspace root")
        .to_path_buf();

    let proto_file = workspace_root.join("proto").join("atlas.proto");

    // Re-run build script only when the proto file changes.
    println!("cargo:rerun-if-changed={}", proto_file.display());

    // Compile .proto → prost_types::FileDescriptorSet (no protoc required).
    let fds = protox::compile([&proto_file], [&workspace_root])?;

    // Generate Rust code from the compiled descriptors.
    prost_build::Config::new().compile_fds(fds)?;

    Ok(())
}
