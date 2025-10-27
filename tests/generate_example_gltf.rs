// skeleton.rs dependencies
use std::path::PathBuf;

use metaverse_mesh::generate::{generate_mesh, generate_skinned_mesh};

#[test]
pub fn generate_example() {
    let mut overalls_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    overalls_path.push("tests/example_json/overalls.json");
    let mut shirt_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    shirt_path.push("tests/example_json/t-shirt.json");
    let mut body_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    body_path.push("tests/example_json/body.json");
    let mut curves_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    curves_path.push("tests/example_json/hair.json");
    let mut avatar_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    avatar_path.push("tests/example_json/avatar.json");

    let mut out_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    out_path.push("tests/generated/Combined.glb");

    let mut out_path_boneless = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    out_path_boneless.push("tests/generated/Boneless.glb");

    let mut out_path_boneless_body = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    out_path_boneless_body.push("tests/generated/BonelessBody.glb");

    generate_skinned_mesh(avatar_path, out_path);
    generate_mesh(overalls_path, out_path_boneless);
    generate_mesh(body_path, out_path_boneless_body);
}
