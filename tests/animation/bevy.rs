use benthic_protocol::default_animations::JointAnimation;
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
    math::Vec3,
    prelude::{AnimationGraph, AnimationGraphHandle, AnimationNodeIndex, Camera3d, Resource},
    scene::SceneRoot,
    transform::components::Transform,
    DefaultPlugins,
};
use reskeletonizer::gltf::export_filtered_animation;
use std::fs::File;
use std::path::PathBuf;

use lazy_static::lazy_static;
use std::collections::HashSet;

lazy_static! {
    static ref PUFFBALL_JOINT_FILTER: HashSet<JointName> = HashSet::from([
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

#[test]
fn display_animation() {
    let mut app = App::new();
    let out_dir = benthic_asset_pipeline::generated_asset_path();
    let path = PathBuf::from(out_dir).join("Animations").join("Stand.json");

    let file = File::open(path).expect("failed to open animation json");
    let animations: Vec<JointAnimation> =
        serde_json::from_reader(file).expect("failed to deserialize animation json");

    let joint_filter: HashSet<_> = animations.iter().map(|j| j.joint).collect();
    let out_path = PathBuf::from("tests/assets/animations/stand_animation.glb");
    export_filtered_animation(&animations, &joint_filter, out_path.clone()).unwrap();

    // Configure WinitPlugin to run on any thread
    app.add_plugins((DefaultPlugins
        .set(WinitPlugin {
            run_on_any_thread: true,
        })
        .set(AssetPlugin {
            file_path: "tests/assets".to_string(),
            mode: AssetMode::Unprocessed,
            ..Default::default()
        })
        .set(ImagePlugin::default_nearest()),));
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
        Transform::from_xyz(0., 0.5, 5.).looking_at(Vec3::new(0., 0.5, 0.), Vec3::Y),
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
            .load(GltfAssetLabel::Animation(0).from_asset("animations/stand_animation.glb")),
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
        SceneRoot(asset_server.load(GltfAssetLabel::Scene(0).from_asset("prod_model.glb"))),
        Transform::from_xyz(0., 0., 0.),
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
