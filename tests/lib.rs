pub mod animation;
pub mod mesh;

use regex::Regex;
use std::{
    fs,
    path::{Path, PathBuf},
};

pub struct ExamplePaths {
    pub overalls: PathBuf,
    pub tshirt: PathBuf,
    pub hair: PathBuf,
    pub body: PathBuf,
    pub avatar: PathBuf,
}

// this replaces the texture paths to the paths of the textures in your local dir.
// this allows the models to be built correctly with links to the correct textures
fn replace_textures_regex(original: &PathBuf, out_dir: &Path) -> PathBuf {
    let json_str = fs::read_to_string(original)
        .unwrap_or_else(|e| panic!("Failed to read {}: {e}", original.display()));
    let re = Regex::new(r"TEXTURE_[A-Za-z0-9_-]+").unwrap();

    let file_path = original
        .parent()
        .unwrap()
        .join(format!(
            "{}.png",
            original
                .file_stem()
                .expect("File has no stem")
                .to_string_lossy()
        ))
        .to_string_lossy()
        .to_string();

    let replaced = re.replace_all(&json_str, file_path);

    let out_path = out_dir.join(original.file_name().unwrap());
    fs::write(&out_path, replaced.as_ref())
        .unwrap_or_else(|e| panic!("Failed to write {}: {e}", out_path.display()));

    out_path
}

pub fn generate_example() -> ExamplePaths {
    //let base_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/example_json");
    let out_json_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/generated/json");
    std::fs::create_dir_all(&out_json_dir).unwrap();

    let mut overalls_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    overalls_path.push("tests/example_json/overalls.json");
    let new_overalls_path = replace_textures_regex(&overalls_path, &out_json_dir);

    let mut shirt_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    shirt_path.push("tests/example_json/t-shirt.json");
    let new_shirt_path = replace_textures_regex(&shirt_path, &out_json_dir);

    let mut body_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    body_path.push("tests/example_json/body.json");
    let new_body_path = replace_textures_regex(&body_path, &out_json_dir);

    let mut curves_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    curves_path.push("tests/example_json/hair.json");
    let new_hair_path = replace_textures_regex(&curves_path, &out_json_dir);

    //let mut button_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    //button_path.push("tests/example_json/button.json");
    //replace_textures_regex(&button_path, &out_json_dir);

    // this is the full avatar json containing all of the sub-outfit pieces
    let mut avatar_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    avatar_path.push("tests/example_json/avatar.json");

    // replace the placeholder text in avatar.json with the actual paths
    let replacements = [
        ("OVERALLS", "overalls.json"),
        ("CYLINDER", "button.json"),
        ("TSHIRT", "t-shirt.json"),
        ("HAIR", "hair.json"),
        ("BODY", "body.json"),
    ];

    let mut avatar_json_str = fs::read_to_string(&avatar_path).unwrap();
    for (placeholder, file_name) in replacements {
        let file_path = out_json_dir.join(file_name).to_string_lossy().to_string();
        avatar_json_str = avatar_json_str.replace(placeholder, &file_path);
    }
    let test_avatar_path = out_json_dir.join("avatar_test.json");
    fs::write(&test_avatar_path, &avatar_json_str).expect("Failed to write avatar_test.json");

    //let mut out_path_button = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    //out_path_button.push("tests/generated/Button.glb");
    //

    // return the paths to all of the cleaned up objects
    ExamplePaths {
        avatar: test_avatar_path,
        overalls: new_overalls_path,
        tshirt: new_shirt_path,
        hair: new_hair_path,
        body: new_body_path,
    }
}
