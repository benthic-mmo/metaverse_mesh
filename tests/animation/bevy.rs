use benthic_protocol::default_animations::{AnimationClip, JointAnimation};
use benthic_protocol::skeleton::JointName;
use bevy::asset::{AssetMode, AssetPlugin};
use bevy::ecs::prelude::*;
use bevy::image::ImagePlugin;
use bevy::light::GlobalAmbientLight;
use bevy::prelude::Name;
use bevy::winit::WinitPlugin;
use bevy::{
    animation::AnimationPlayer,
    app::{App, PluginGroup, Startup},
    asset::{AssetServer, Assets, Handle},
    color::Color,
    ecs::system::{Commands, Query, Res, ResMut},
    gltf::GltfAssetLabel,
    math::{Dir3, Vec3}, // Added Dir3 import
    prelude::{AnimationGraph, AnimationGraphHandle, AnimationNodeIndex, Camera3d, Resource},
    transform::components::Transform,
    DefaultPlugins,
};
use bevy_panorbit_camera::PanOrbitCamera;
use bevy_world_serialization::WorldAssetRoot; // Added WorldAssetRoot import
use lazy_static::lazy_static;
use metaverse_mesh::animation::gltf::export_filtered_animation;
use metaverse_mesh::mesh::generate::generate_skinned_mesh;
use regex::Regex;
use std::collections::BTreeSet;
use std::fs::{self};
use std::path::PathBuf;

lazy_static! {
    static ref PUFFBALL_JOINT_FILTER: BTreeSet<JointName> = BTreeSet::from([
        JointName::Pelvis,
        JointName::Torso,
        JointName::Tail1,
        JointName::HipLeft,
        JointName::HipRight,
        JointName::Chest,
        JointName::Neck,
        JointName::CollarLeft,
        JointName::CollarRight,
        JointName::Head,
        JointName::Skull,
        JointName::FaceRoot,
        JointName::FaceJaw,
        JointName::FaceJawShaper,
        JointName::FaceEar1Left,
        JointName::FaceEar1Right,
        JointName::FaceEar2Left,
        JointName::FaceEar2Right,
        JointName::ShoulderLeft,
        JointName::ElbowLeft,
        JointName::WristLeft,
        JointName::HandIndex1Left,
        JointName::HandMiddle1Left,
        JointName::HandRing1Left,
        JointName::HandPinky1Left,
        JointName::HandThumb1Left,
        JointName::HandIndex2Left,
        JointName::HandMiddle2Left,
        JointName::HandRing2Left,
        JointName::HandPinky2Left,
        JointName::HandThumb2Left,
        JointName::HandIndex3Left,
        JointName::HandMiddle3Left,
        JointName::HandRing3Left,
        JointName::HandPinky3Left,
        JointName::HandThumb3Left,
        JointName::ShoulderRight,
        JointName::ElbowRight,
        JointName::WristRight,
        JointName::HandIndex1Right,
        JointName::HandMiddle1Right,
        JointName::HandRing1Right,
        JointName::HandPinky1Right,
        JointName::HandThumb1Right,
        JointName::HandIndex2Right,
        JointName::HandMiddle2Right,
        JointName::HandRing2Right,
        JointName::HandPinky2Right,
        JointName::HandThumb2Right,
        JointName::HandIndex3Right,
        JointName::HandMiddle3Right,
        JointName::HandRing3Right,
        JointName::HandPinky3Right,
        JointName::HandThumb3Right,
        JointName::Tail1,
        JointName::Tail2,
        JointName::Tail3,
        JointName::Tail4,
        JointName::Tail5,
        JointName::Tail6,
        JointName::KneeLeft,
        JointName::AnkleLeft,
        JointName::FootLeft,
        JointName::KneeRight,
        JointName::AnkleRight,
        JointName::FootRight,
    ]);
}
fn generated_animation_path(name: &str) -> PathBuf {
    let path = PathBuf::from("tests").join("animation").join("generated");

    std::fs::create_dir_all(&path).unwrap();

    path.join(name)
}

fn load_animation(name: &str) -> AnimationClip {
    let path = benthic_asset_pipeline::generated_asset_path();

    let file = std::fs::File::open(
        PathBuf::from(path)
            .join("Animations")
            .join(format!("{name}.json")),
    )
    .unwrap();

    serde_json::from_reader(file).expect("failed to deserialize animation json")
}

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

pub fn generate_example() {
    let out_json_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/mesh/generated/json");
    std::fs::create_dir_all(&out_json_dir).unwrap();

    let mut overalls_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    overalls_path.push("tests/mesh/example_json/overalls.json");

    let mut shirt_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    shirt_path.push("tests/mesh/example_json/t-shirt.json");
    replace_textures_regex(&shirt_path, &out_json_dir);

    let mut body_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    body_path.push("tests/mesh/example_json/body.json");

    let mut curves_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    curves_path.push("tests/mesh/example_json/hair.json");
    replace_textures_regex(&curves_path, &out_json_dir);

    //let mut button_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    //button_path.push("tests/mesh/example_json/button.json");
    //replace_textures_regex(&button_path, &out_json_dir);

    // this is the full avatar json containing all of the sub-outfit pieces
    let mut avatar_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    avatar_path.push("tests/mesh/example_json/avatar.json");

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
    out_path.push("tests/mesh/generated/combined.glb");

    //let mut out_path_button = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    //out_path_button.push("tests/mesh/generated/Button.glb");

    generate_skinned_mesh(test_avatar_path, out_path).unwrap();
}

#[test]
fn display_animation() {
    let mut app = App::new();

    //let animations = load_animation("Stand_Correct");
    let animations = load_animation("Stand_Correct");

    let mut joint_filter = BTreeSet::new();
    joint_filter.extend(PUFFBALL_JOINT_FILTER.iter().copied());

    let out_path = generated_animation_path("puffball.glb");

    export_filtered_animation(&animations, &joint_filter, out_path).unwrap();
    generate_example();

    // Configure WinitPlugin to run on any thread
    app.add_plugins((DefaultPlugins
        .set(WinitPlugin {
            run_on_any_thread: true,
        })
        .set(AssetPlugin {
            file_path: "tests".to_string(),
            mode: AssetMode::Unprocessed,
            ..Default::default()
        })
        .set(ImagePlugin::default_nearest()),));
    app.add_plugins(bevy_panorbit_camera::PanOrbitCameraPlugin);
    app.finish();
    app.cleanup();
    // Resources
    app.insert_resource(GlobalAmbientLight {
        brightness: 600.,
        color: Color::WHITE,
        affects_lightmapped_meshes: true,
    });

    // Systems
    app.add_systems(Startup, spawn_camera)
        .add_systems(Startup, spawn_models)
        .add_systems(Startup, setup_animation_graph);

    // Observers
    app.add_observer(animation_player_added);

    app.run();
}

#[derive(Debug, Resource)]
struct AnimationGraphCache {
    animations: Vec<AnimationNodeIndex>,
    graph: Handle<AnimationGraph>,
}

fn spawn_camera(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(1.5, 1.5, 3.5).looking_at(Vec3::new(0.0, 0.0, 0.0), Dir3::Y),
        PanOrbitCamera {
            focus: Vec3::new(0.0, 0.0, 0.0),
            orbit_smoothness: 0.1,
            pan_smoothness: 0.1,
            zoom_smoothness: 0.1,
            ..Default::default()
        },
    ));
}

fn setup_animation_graph(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut graphs: ResMut<Assets<AnimationGraph>>,
) {
    let mut graph = AnimationGraph::new();
    let animations = vec![graph.add_clip(
        asset_server
            .load(GltfAssetLabel::Animation(0).from_asset("animation/generated/puffball.glb")),
        1.0,
        graph.root,
    )];

    let graph_handle = graphs.add(graph);
    commands.insert_resource(AnimationGraphCache {
        animations,
        graph: graph_handle,
    });
}

fn spawn_models(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.spawn((
        WorldAssetRoot(
            asset_server.load(GltfAssetLabel::Scene(0).from_asset("mesh/generated/combined.glb")),
        ),
        Transform::from_xyz(0., -1.0, 0.),
        Name::new("Puffball"),
    ));
}

fn animation_player_added(
    trigger: On<Add, AnimationPlayer>,
    mut commands: Commands,
    graph_cache: Res<AnimationGraphCache>,
    mut players: Query<&mut AnimationPlayer>,
) {
    if let Ok(mut player) = players.get_mut(trigger.entity) {
        player.play(graph_cache.animations[0]).repeat();
        commands
            .entity(trigger.entity)
            .insert(AnimationGraphHandle(graph_cache.graph.clone()));
    }
}
