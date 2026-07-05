use benthic_protocol::default_animations::AnimationClip;
use benthic_protocol::skeleton::JointName;
use bevy::asset::{AssetMode, AssetPlugin};
use bevy::ecs::prelude::*;
use bevy::image::ImagePlugin;
use bevy::light::GlobalAmbientLight;
use bevy::prelude::Name;
use bevy::winit::WinitPlugin;
use bevy::{
    DefaultPlugins,
    animation::AnimationPlayer,
    app::{App, PluginGroup, Startup},
    asset::{AssetServer, Assets, Handle},
    color::Color,
    ecs::system::{Commands, Query, Res, ResMut},
    gltf::GltfAssetLabel,
    math::{Dir3, Vec3}, // Added Dir3 import
    prelude::{AnimationGraph, AnimationGraphHandle, AnimationNodeIndex, Camera3d, Resource},
    transform::components::Transform,
};
use bevy_panorbit_camera::PanOrbitCamera;
use bevy_world_serialization::WorldAssetRoot; // Added WorldAssetRoot import
use lazy_static::lazy_static;
use metaverse_mesh::animation::gltf::export_filtered_animation;
use metaverse_mesh::mesh::generate::generate_skinned_mesh;
use std::collections::BTreeSet;
use std::path::PathBuf;

use crate::generate_example;

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
    let filename = path.join("Animations").join(format!("{name}.json"));

    println!("loading animation: {:?}", filename);
    let file = std::fs::File::open(filename).unwrap();

    serde_json::from_reader(file).expect("failed to deserialize animation json")
}

#[test]
fn run_stand_only() {
    display_animation("Stand");
}

#[test]
fn run_stand_correct() {
    display_animation("Stand_Correct");
}

#[test]
fn run_standy() {
    display_animation("standy");
}

fn display_animation(animation: &str) {
    let mut app = App::new();

    let animations = load_animation(animation);

    let mut joint_filter = BTreeSet::new();
    joint_filter.extend(PUFFBALL_JOINT_FILTER.iter().copied());

    let out_path = generated_animation_path("puffball.glb");

    export_filtered_animation(&animations, &joint_filter, out_path.clone()).unwrap();

    let generated_paths = generate_example();
    let mut mesh_out_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    mesh_out_path.push("tests/generated/animation_combined.glb");

    generate_skinned_mesh(generated_paths.avatar.clone(), mesh_out_path.clone()).unwrap_or_else(
        |e| {
            panic!(
                "generate_skinned_mesh failed\n  input: {:?}\n  output: {:?}\n  error: {e:?}",
                generated_paths.avatar, out_path
            )
        },
    );

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
    let animations = vec![
        graph.add_clip(
            asset_server
                .load(GltfAssetLabel::Animation(0).from_asset("animation/generated/puffball.glb")),
            1.0,
            graph.root,
        ),
    ];

    let graph_handle = graphs.add(graph);
    commands.insert_resource(AnimationGraphCache {
        animations,
        graph: graph_handle,
    });
}

fn spawn_models(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.spawn((
        WorldAssetRoot(
            asset_server
                .load(GltfAssetLabel::Scene(0).from_asset("generated/animation_combined.glb")),
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
