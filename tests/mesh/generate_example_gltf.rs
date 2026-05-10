// skeleton.rs dependencies
use bevy::prelude::{PluginGroup, Startup};
use bevy::{
    app::App,
    asset::{AssetMode, AssetPlugin, AssetServer},
    camera::Camera3d,
    ecs::{
        name::Name,
        system::{Commands, Res},
    },
    gltf::GltfAssetLabel,
    light::DirectionalLight,
    scene::SceneRoot,
    transform::components::Transform,
    winit::WinitPlugin,
    DefaultPlugins,
};
use glam::Vec3;
use metaverse_mesh::generate::{generate_mesh, generate_skinned_mesh};
use regex::Regex;
use std::{fs, path::PathBuf};

fn replace_textures_regex(original: &PathBuf, out_dir: &PathBuf) -> PathBuf {
    let json_str =
        fs::read_to_string(original).expect(&format!("Failed to read {}", original.display()));

    let re = Regex::new(r"TEXTURE_[A-Z0-9_]+").unwrap();

    let replaced = re.replace_all(&json_str, {
        let file_name = format!(
            "{}.png",
            original
                .file_stem()
                .expect("File has no stem")
                .to_string_lossy()
        );
        original
            .parent()
            .unwrap()
            .join(file_name.to_string())
            .to_string_lossy()
            .to_string()
    });

    let out_path = out_dir.join(original.file_name().unwrap());
    fs::write(&out_path, replaced.as_ref())
        .expect(&format!("Failed to write {}", out_path.display()));
    out_path
}

#[test]
pub fn generate_example() {
    //let base_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/example_json");
    let out_json_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/generated/json");
    std::fs::create_dir_all(&out_json_dir).unwrap();

    let mut overalls_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    overalls_path.push("tests/example_json/overalls.json");
    let new_overalls_path = replace_textures_regex(&overalls_path, &out_json_dir);

    let mut shirt_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    shirt_path.push("tests/example_json/t-shirt.json");
    replace_textures_regex(&shirt_path, &out_json_dir);

    let mut body_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    body_path.push("tests/example_json/body.json");
    let new_body_path = replace_textures_regex(&body_path, &out_json_dir);

    let mut curves_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    curves_path.push("tests/example_json/hair.json");
    replace_textures_regex(&curves_path, &out_json_dir);

    let mut button_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    button_path.push("tests/example_json/button.json");
    replace_textures_regex(&button_path, &out_json_dir);

    // this is the full avatar json containing all of the sub-outfit pieces
    let mut avatar_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    avatar_path.push("tests/example_json/avatar.json");

    let replacements = [
        ("OVERALLS", "overalls.json"),
        ("CYLINDER", "body.json"),
        ("TSHIRT", "t-shirt.json"),
        ("CURVES", "hair.json"),
        ("BODY", "body.json"),
    ];

    let mut avatar_json_str = fs::read_to_string(&avatar_path).unwrap();
    for (placeholder, file_name) in replacements {
        let file_path = out_json_dir.join(file_name).to_string_lossy().to_string();
        avatar_json_str = avatar_json_str.replace(placeholder, &file_path);
    }
    let test_avatar_path = out_json_dir.join("avatar_test.json");
    fs::write(&test_avatar_path, &avatar_json_str).expect("Failed to write avatar_test.json");

    let mut out_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    out_path.push("tests/generated/Combined.glb");

    let mut out_path_boneless = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    out_path_boneless.push("tests/generated/Boneless.glb");

    let mut out_path_boneless_body = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    out_path_boneless_body.push("tests/generated/BonelessBody.glb");

    let mut out_path_button = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    out_path_button.push("tests/generated/Button.glb");

    generate_skinned_mesh(test_avatar_path, out_path).unwrap();
    generate_mesh(new_overalls_path, out_path_boneless).unwrap();
    generate_mesh(new_body_path, out_path_boneless_body).unwrap();
}

#[test]
fn display_generated_models() {
    let mut app = App::new();

    // Point Bevy to your generated assets folder

    app.add_plugins(
        DefaultPlugins
            .set(WinitPlugin {
                run_on_any_thread: true,
            })
            .set(AssetPlugin {
                file_path: "tests/generated".to_string(),
                mode: AssetMode::Unprocessed,
                ..Default::default()
            }),
    );

    app.add_systems(Startup, setup);

    app.run();
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    // Camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 1.0, 5.0).looking_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y),
    ));

    // Light
    commands.spawn((
        DirectionalLight::default(),
        Transform::from_xyz(3.0, 5.0, 3.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // Load your generated GLBs
    commands.spawn((
        SceneRoot(asset_server.load(GltfAssetLabel::Scene(0).from_asset("Combined.glb"))),
        Transform::from_xyz(0.0, 0.0, 0.0),
        Name::new("Combined"),
    ));

    commands.spawn((
        SceneRoot(asset_server.load(GltfAssetLabel::Scene(0).from_asset("Boneless.glb"))),
        Transform::from_xyz(-2.0, 0.0, 0.0),
        Name::new("Boneless"),
    ));

    commands.spawn((
        SceneRoot(asset_server.load(GltfAssetLabel::Scene(0).from_asset("BonelessBody.glb"))),
        Transform::from_xyz(2.0, 0.0, 0.0),
        Name::new("BonelessBody"),
    ));
}
