use std::path::PathBuf;

use bevy::prelude::{Dir3, PluginGroup, Startup, Vec3};
use bevy::{
    DefaultPlugins,
    app::App,
    asset::{AssetMode, AssetPlugin, AssetServer},
    camera::Camera3d,
    ecs::{
        name::Name,
        system::{Commands, Res},
    },
    gltf::GltfAssetLabel,
    light::DirectionalLight,
    transform::components::Transform,
    winit::WinitPlugin,
};
use bevy_world_serialization::WorldAssetRoot;
use metaverse_mesh::mesh::generate::{generate_mesh, generate_skinned_mesh};

use crate::generate_example;

#[test]
fn test_generate_example() {
    let mut out_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    out_path.push("tests/mesh/generated/mesh_combined.glb");

    let mut out_path_boneless = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    out_path_boneless.push("tests/mesh/generated/Boneless.glb");

    let mut out_path_boneless_body = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    out_path_boneless_body.push("tests/mesh/generated/BonelessBody.glb");

    let example_paths = generate_example();
    generate_skinned_mesh(example_paths.avatar, out_path).unwrap();
    generate_mesh(example_paths.overalls, out_path_boneless).unwrap();
    generate_mesh(example_paths.body, out_path_boneless_body).unwrap();
}

#[test]
fn display_generated_models() {
    let mut app = App::new();
    app.add_plugins(
        DefaultPlugins
            .set(WinitPlugin {
                run_on_any_thread: true,
            })
            .set(AssetPlugin {
                file_path: "tests/mesh/generated".to_string(),
                mode: AssetMode::Unprocessed,
                ..Default::default()
            }),
    );

    app.add_systems(Startup, setup);
    app.run();
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 1.0, 5.0).looking_at(Vec3::new(0.0, 1.0, 0.0), Dir3::Y),
    ));

    commands.spawn((
        DirectionalLight::default(),
        Transform::from_xyz(3.0, 5.0, 3.0).looking_at(Vec3::ZERO, Dir3::Y),
    ));

    commands.spawn((
        WorldAssetRoot(asset_server.load(GltfAssetLabel::Scene(0).from_asset("mesh_combined.glb"))),
        Transform::from_xyz(0.0, 0.0, 0.0),
        Name::new("Combined"),
    ));

    commands.spawn((
        WorldAssetRoot(asset_server.load(GltfAssetLabel::Scene(0).from_asset("Boneless.glb"))),
        Transform::from_xyz(-2.0, 0.0, 0.0),
        Name::new("Boneless"),
    ));

    commands.spawn((
        WorldAssetRoot(asset_server.load(GltfAssetLabel::Scene(0).from_asset("BonelessBody.glb"))),
        Transform::from_xyz(2.0, 0.0, 0.0),
        Name::new("BonelessBody"),
    ));
}
