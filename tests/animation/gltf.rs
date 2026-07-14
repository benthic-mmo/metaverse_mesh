use benthic_protocol::default_animations::JointAnimation;
use benthic_protocol::skeleton::JointName;
use metaverse_mesh::animation::gltf::export_filtered_animation;
use std::collections::BTreeSet;
use std::path::PathBuf;

use lazy_static::lazy_static;

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
fn load_animation(name: &str) -> Vec<JointAnimation> {
    let path = benthic_asset_pipeline::generated_asset_path();

    let file = std::fs::File::open(
        PathBuf::from(path)
            .join("Animations")
            .join(format!("{name}.json")),
    )
    .unwrap();

    serde_json::from_reader(file).expect("failed to deserialize animation json")
}

fn generated_animation_path(name: &str) -> PathBuf {
    let path = PathBuf::from("tests").join("animation").join("generated");

    std::fs::create_dir_all(&path).unwrap();

    path.join(name)
}

#[test]
fn filter_skeleton() {
    let animations = load_animation("Stand");

    if let Some(_j) = animations.iter().find(|a| a.joint == JointName::Pelvis) {
        //println!("{:?}", j);
    }
}

#[test]
fn build_all_bones_gltf() {
    let animations = load_animation("Stand");

    let joint_filter: BTreeSet<_> = animations.iter().map(|j| j.joint).collect();

    let out_path = generated_animation_path("stand_animation.glb");

    export_filtered_animation(&animations, &joint_filter, out_path).unwrap();
}

#[test]
fn build_bow() {
    let animations = load_animation("Bow");

    let joint_filter: BTreeSet<_> = animations.iter().map(|j| j.joint).collect();

    let out_path = generated_animation_path("bow_animation.glb");

    export_filtered_animation(&animations, &joint_filter, out_path).unwrap();
}

#[test]
fn build_only_arm() {
    let animations = load_animation("Stand");

    let mut joint_filter = BTreeSet::new();
    joint_filter.insert(JointName::ShoulderLeft);
    joint_filter.insert(JointName::ElbowLeft);

    let out_path = generated_animation_path("arm.glb");

    export_filtered_animation(&animations, &joint_filter, out_path).unwrap();
}

#[test]
fn build_only_puffball() {
    let animations = load_animation("Stand");

    let mut joint_filter = BTreeSet::new();
    joint_filter.extend(PUFFBALL_JOINT_FILTER.iter().copied());

    let out_path = generated_animation_path("puffball.glb");

    export_filtered_animation(&animations, &joint_filter, out_path).unwrap();
}

#[test]
fn build_only_puffball_bvh() {
    let animations = load_animation("Stand");

    let mut joint_filter = BTreeSet::new();
    joint_filter.extend(PUFFBALL_JOINT_FILTER.iter().copied());

    let out_path = generated_animation_path("puffball_bow.glb");

    export_filtered_animation(&animations, &joint_filter, out_path).unwrap();
}
