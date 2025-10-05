// skeleton.rs dependencies
use metaverse_gltf::skinned_mesh::generate_skinned_mesh;
use std::path::PathBuf;

#[test]
pub fn generate_example() {
    let mut paths = Vec::new();
    let mut skeleton_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    skeleton_path.push("tests/example_json/skeleton.json");

    let mut overalls_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    overalls_path.push("tests/example_json/overalls.json");
    let mut shirt_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    shirt_path.push("tests/example_json/t-shirt.json");
    let mut body_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    body_path.push("tests/example_json/body.json");
    let mut curves_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    curves_path.push("tests/example_json/curves.json");

    let mut out_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    out_path.push("tests/generated/Combined.glb");

    paths.push(overalls_path);
    paths.push(shirt_path);
    paths.push(body_path);
    paths.push(curves_path);

    generate_skinned_mesh(paths, skeleton_path, out_path);
}
