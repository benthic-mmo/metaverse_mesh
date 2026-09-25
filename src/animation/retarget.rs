use crate::errors::MetaverseMeshAnimationError as AnimationError;
use benthic_protocol::{
    default_animations::{AnimationClip, JointAnimation},
    skeleton::{JointName, Skeleton},
};
use glam::Mat4;
use indexmap::IndexMap;
use std::path::Path;
use std::{
    collections::{BTreeSet, HashSet},
    fs,
};

fn effective_parent(
    skeleton: &Skeleton,
    joint_name: JointName,
    joint_filter: &BTreeSet<JointName>,
) -> Result<Option<JointName>, AnimationError> {
    let joint = skeleton
        .joints
        .get(&joint_name)
        .ok_or(AnimationError::MissingJoint { joint: joint_name })?;

    let mut parent = joint.parent;

    while let Some(parent_name) = parent {
        if joint_filter.contains(&parent_name) {
            return Ok(Some(parent_name));
        }

        parent = skeleton
            .joints
            .get(&parent_name)
            .ok_or(AnimationError::MissingJoint { joint: parent_name })?
            .parent;
    }

    Ok(None)
}

fn joint_global_transform(
    skeleton: &Skeleton,
    joint_name: JointName,
) -> Result<Mat4, AnimationError> {
    skeleton
        .joints
        .get(&joint_name)
        .ok_or(AnimationError::MissingJoint { joint: joint_name })?
        .global_transforms
        .last()
        .map(|transform| transform.transform)
        .ok_or(AnimationError::MissingTransform {
            joint: joint_name,
            transform_type: "global",
        })
}

fn joint_local_transform(
    skeleton: &Skeleton,
    joint_name: JointName,
) -> Result<Mat4, AnimationError> {
    skeleton
        .joints
        .get(&joint_name)
        .ok_or(AnimationError::MissingJoint { joint: joint_name })?
        .local_transforms
        .last()
        .map(|transform| transform.transform)
        .ok_or(AnimationError::MissingTransform {
            joint: joint_name,
            transform_type: "local",
        })
}

pub fn filter_bind_skeleton(
    skeleton: &Skeleton,
    used_joints: &BTreeSet<JointName>,
) -> Result<Skeleton, AnimationError> {
    let root = skeleton
        .root
        .iter()
        .copied()
        .filter(|joint| used_joints.contains(joint))
        .collect();

    let mut filtered_skeleton = Skeleton {
        root,
        joints: IndexMap::new(),
    };

    for (&joint_name, joint) in &skeleton.joints {
        if !used_joints.contains(&joint_name) {
            continue;
        }

        let parent = effective_parent(skeleton, joint_name, used_joints)?;
        let joint_global = joint_global_transform(skeleton, joint_name)?;

        let parent_global = match parent {
            Some(parent) => joint_global_transform(skeleton, parent)?,
            None => Mat4::IDENTITY,
        };

        let mut filtered_joint = joint.clone();

        filtered_joint.parent = parent;
        filtered_joint.children.clear();

        filtered_joint
            .local_transforms
            .last_mut()
            .ok_or(AnimationError::MissingTransform {
                joint: joint_name,
                transform_type: "local",
            })?
            .transform = parent_global.inverse() * joint_global;

        filtered_joint
            .global_transforms
            .last_mut()
            .ok_or(AnimationError::MissingTransform {
                joint: joint_name,
                transform_type: "global",
            })?
            .transform = joint_global;

        filtered_skeleton.joints.insert(joint_name, filtered_joint);
    }

    let joints = filtered_skeleton.joints.clone();

    for (&joint_name, joint) in &joints {
        if let Some(parent) = joint.parent {
            let parent_joint = filtered_skeleton
                .joints
                .get_mut(&parent)
                .ok_or(AnimationError::MissingJoint { joint: parent })?;

            parent_joint.children.push(joint_name);
        }
    }

    Ok(filtered_skeleton)
}

pub fn filter_animation_keyframes(
    animations: &[JointAnimation],
    skeleton: &Skeleton,
    used_joints: &BTreeSet<JointName>,
) -> Result<Vec<JointAnimation>, AnimationError> {
    let mut filtered = Vec::new();

    for joint_anim in animations {
        if !used_joints.contains(&joint_anim.joint) {
            continue;
        }

        let joint = skeleton
            .joints
            .get(&joint_anim.joint)
            .ok_or(AnimationError::MissingJoint {
                joint: joint_anim.joint,
            })?;

        let original_parent_global = match joint.parent {
            Some(parent) => joint_global_transform(skeleton, parent)?,
            None => Mat4::IDENTITY,
        };

        let effective_parent_global =
            match effective_parent(skeleton, joint_anim.joint, used_joints)? {
                Some(parent) => joint_global_transform(skeleton, parent)?,
                None => Mat4::IDENTITY,
            };

        let parent_conversion = effective_parent_global.inverse() * original_parent_global;

        let mut animation = joint_anim.clone();

        for keyframe in &mut animation.translations {
            let local = Mat4::from_translation(keyframe.value);

            keyframe.value = (parent_conversion * local)
                .to_scale_rotation_translation()
                .2;
        }

        for keyframe in &mut animation.rotations {
            let local = Mat4::from_quat(keyframe.value);

            keyframe.value = (parent_conversion * local)
                .to_scale_rotation_translation()
                .1;
        }

        for keyframe in &mut animation.scales {
            let local = Mat4::from_scale(keyframe.value);

            keyframe.value = (parent_conversion * local)
                .to_scale_rotation_translation()
                .0;
        }

        filtered.push(animation);
    }

    Ok(filtered)
}

pub fn apply_joint_scale(
    mut animation: AnimationClip,
    animation_out_path: &Path,
    target_skeleton: &Skeleton,
    used_joints: &BTreeSet<JointName>,
) -> Result<(), AnimationError> {
    let target_skeleton = &filter_bind_skeleton(target_skeleton, used_joints)?;

    for joint_animation in &mut animation.joints {
        let joint = joint_animation.joint;

        let Some(source_joint) = animation.bind_skeleton.joints.get(&joint) else {
            continue;
        };

        let Some(target_joint) = target_skeleton.joints.get(&joint) else {
            continue;
        };

        let source_bind_local = joint_local_transform(&animation.bind_skeleton, joint)?;

        let target_bind_local = joint_local_transform(target_skeleton, joint)?;

        let source_parent_world = match source_joint.parent {
            Some(parent) => joint_global_transform(&animation.bind_skeleton, parent)?,
            None => Mat4::IDENTITY,
        };

        let source_bind_world = source_parent_world * source_bind_local;

        let inverse_source_bind = source_bind_world.inverse();

        let target_parent_world = match target_joint.parent {
            Some(parent) => joint_global_transform(target_skeleton, parent)?,
            None => Mat4::IDENTITY,
        };

        let target_bind_world = target_parent_world * target_bind_local;

        let inverse_target_parent = target_parent_world.inverse();

        let (bind_scale, bind_rotation, bind_translation) =
            source_bind_local.to_scale_rotation_translation();

        for frame in &mut joint_animation.translations {
            let local =
                Mat4::from_scale_rotation_translation(bind_scale, bind_rotation, frame.value);

            let source_world = source_parent_world * local;
            let delta = inverse_source_bind * source_world;

            let target_local = inverse_target_parent * target_bind_world * delta;

            frame.value = target_local.to_scale_rotation_translation().2;
        }

        for frame in &mut joint_animation.rotations {
            let local =
                Mat4::from_scale_rotation_translation(bind_scale, frame.value, bind_translation);

            let source_world = source_parent_world * local;
            let delta = inverse_source_bind * source_world;

            let target_local = inverse_target_parent * target_bind_world * delta;

            frame.value = target_local.to_scale_rotation_translation().1;
        }

        for frame in &mut joint_animation.scales {
            let local =
                Mat4::from_scale_rotation_translation(frame.value, bind_rotation, bind_translation);

            let source_world = source_parent_world * local;
            let delta = inverse_source_bind * source_world;

            let target_local = inverse_target_parent * target_bind_world * delta;

            frame.value = target_local.to_scale_rotation_translation().0;
        }
    }

    animation.bind_skeleton = target_skeleton.clone();

    let file =
        fs::File::create(animation_out_path).map_err(|source| AnimationError::CreateOutput {
            path: animation_out_path.to_path_buf(),
            source,
        })?;

    serde_json::to_writer_pretty(file, &animation).map_err(|source| AnimationError::Serialize {
        path: animation_out_path.to_path_buf(),
        source,
    })?;

    Ok(())
}

pub fn check_skeleton_cycles(skeleton: &Skeleton) -> Result<(), AnimationError> {
    fn visit(
        joint: JointName,
        skeleton: &Skeleton,
        visiting: &mut HashSet<JointName>,
        visited: &mut HashSet<JointName>,
    ) -> Result<(), AnimationError> {
        if visiting.contains(&joint) {
            return Err(AnimationError::SkeletonCycle { joint });
        }

        if visited.contains(&joint) {
            return Ok(());
        }

        visiting.insert(joint);

        if let Some(node) = skeleton.joints.get(&joint) {
            for child in &node.children {
                visit(*child, skeleton, visiting, visited)?;
            }
        }

        visiting.remove(&joint);
        visited.insert(joint);

        Ok(())
    }

    let mut visiting = HashSet::new();
    let mut visited = HashSet::new();

    for joint in skeleton.joints.keys() {
        visit(*joint, skeleton, &mut visiting, &mut visited)?;
    }

    Ok(())
}
