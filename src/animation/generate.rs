use crate::animation::retarget::{
    apply_joint_scale, filter_animation_keyframes, filter_bind_skeleton,
};
use crate::errors::MetaverseMeshError;
use crate::{animation::gltf::export_animation, errors::MetaverseMeshAnimationError};
use benthic_protocol::default_animations::AnimationClip;
use benthic_protocol::skeleton::{JointName, Skeleton};
use default_asset_converter::generated::DEFAULT_SKELETON;
use std::collections::BTreeSet;
use std::path::Path;
use std::{
    ffi::{CStr, c_char},
    fs,
    path::PathBuf,
};

pub fn generate_gltf_animation(
    animation_json_path: &PathBuf,
    out_path: &PathBuf,
) -> Result<(), MetaverseMeshError> {
    let file = fs::File::open(animation_json_path)
        .unwrap_or_else(|e| panic!("Failed to read {:?}, {:?}", animation_json_path, e));
    let animations: AnimationClip = serde_json::from_reader(&file)
        .unwrap_or_else(|e| panic!("Failed to deserialize joint animation {:?}", e));
    export_animation(&animations, out_path)
}

fn load_animation(path: &Path) -> Result<AnimationClip, MetaverseMeshAnimationError> {
    let file = fs::File::open(path).map_err(|source| MetaverseMeshAnimationError::ReadFile {
        path: path.to_path_buf(),
        source,
    })?;

    serde_json::from_reader(file).map_err(|source| MetaverseMeshAnimationError::Deserialize {
        path: path.to_path_buf(),
        source,
    })
}

pub fn retarget_gltf_animation<S>(
    animation_json_path: &Path,
    target_skeleton: S,
    filtered_animation_out_path: &Path,
    out_path: &Path,
) -> Result<(), MetaverseMeshAnimationError>
where
    S: IntoSkeleton,
{
    let target_skeleton = target_skeleton.into_skeleton()?;
    let used_joints = used_joints(&target_skeleton);

    let animation = load_animation(animation_json_path)?;
    let skeleton = &DEFAULT_SKELETON;

    let animation_clip = AnimationClip {
        bind_skeleton: filter_bind_skeleton(skeleton, &used_joints)?,
        joints: filter_animation_keyframes(&animation.joints, skeleton, &used_joints)?,
    };
    let file = fs::File::create(filtered_animation_out_path).map_err(|source| {
        MetaverseMeshAnimationError::CreateOutput {
            path: filtered_animation_out_path.to_path_buf(),
            source,
        }
    })?;

    serde_json::to_writer_pretty(file, &animation_clip).map_err(|source| {
        MetaverseMeshAnimationError::Serialize {
            path: filtered_animation_out_path.to_path_buf(),
            source,
        }
    })?;
    apply_joint_scale(animation_clip, out_path, &target_skeleton, &used_joints)
}

pub fn retarget_filtered_gltf_animation<S>(
    filtered_animation_json_path: &Path,
    target_skeleton: S,
    out_path: &Path,
) -> Result<(), MetaverseMeshAnimationError>
where
    S: IntoSkeleton,
{
    let target_skeleton = target_skeleton.into_skeleton()?;
    let animation = load_animation(filtered_animation_json_path)?;
    let used_joints = used_joints(&target_skeleton);

    apply_joint_scale(animation, out_path, &target_skeleton, &used_joints)
}

#[unsafe(no_mangle)]
/// Allow external projects to generate animation from json. This will return the string of where the file
/// was generated on disk!
pub unsafe extern "C" fn generate_gltf_animation_legacy(
    animation_json_path: *const c_char,
    out_path: *const c_char,
) -> *mut c_char {
    // Parse animation json path
    let animation_json = {
        let s = unsafe {
            CStr::from_ptr(animation_json_path)
                .to_string_lossy()
                .into_owned()
        };

        PathBuf::from(s)
    };

    // Parse output path
    let out = {
        let s = unsafe { CStr::from_ptr(out_path).to_string_lossy().into_owned() };

        PathBuf::from(s)
    };

    // Generate animation
    match generate_gltf_animation(&animation_json, &out) {
        Ok(_) => std::ffi::CString::new("Success").unwrap().into_raw(),

        Err(e) => {
            eprintln!("Failed to generate joint animation: {:?}", e);
            std::ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
/// Retarget a GLTF animation using a target skeleton loaded from JSON.
///
/// Returns "Success" on success or null on failure.
pub unsafe extern "C" fn retarget_gltf_animation_legacy(
    animation_json_path: *const c_char,
    target_skeleton_path: *const c_char,
    filtered_animation_out_path: *const c_char,
    out_path: *const c_char,
) -> *mut c_char {
    let animation_json = {
        let s = unsafe {
            CStr::from_ptr(animation_json_path)
                .to_string_lossy()
                .into_owned()
        };

        PathBuf::from(s)
    };

    let target_skeleton = {
        let s = unsafe {
            CStr::from_ptr(target_skeleton_path)
                .to_string_lossy()
                .into_owned()
        };

        PathBuf::from(s)
    };

    let filtered_animation_out = {
        let s = unsafe {
            CStr::from_ptr(filtered_animation_out_path)
                .to_string_lossy()
                .into_owned()
        };

        PathBuf::from(s)
    };

    let out = {
        let s = unsafe { CStr::from_ptr(out_path).to_string_lossy().into_owned() };

        PathBuf::from(s)
    };

    match retarget_gltf_animation(
        &animation_json,
        target_skeleton.as_path(),
        &filtered_animation_out,
        &out,
    ) {
        Ok(_) => std::ffi::CString::new("Success").unwrap().into_raw(),

        Err(e) => {
            eprintln!("Failed to retarget GLTF animation: {:?}", e);
            std::ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
/// Retarget a previously filtered GLTF animation using a target skeleton
/// loaded from JSON.
///
/// Returns "Success" on success or null on failure.
pub unsafe extern "C" fn retarget_filtered_gltf_animation_legacy(
    filtered_animation_json_path: *const c_char,
    target_skeleton_path: *const c_char,
    out_path: *const c_char,
) -> *mut c_char {
    let filtered_animation_json = {
        let s = unsafe {
            CStr::from_ptr(filtered_animation_json_path)
                .to_string_lossy()
                .into_owned()
        };

        PathBuf::from(s)
    };

    let target_skeleton = {
        let s = unsafe {
            CStr::from_ptr(target_skeleton_path)
                .to_string_lossy()
                .into_owned()
        };

        PathBuf::from(s)
    };

    let out = {
        let s = unsafe { CStr::from_ptr(out_path).to_string_lossy().into_owned() };

        PathBuf::from(s)
    };

    match retarget_filtered_gltf_animation(
        &filtered_animation_json,
        target_skeleton.as_path(),
        &out,
    ) {
        Ok(_) => std::ffi::CString::new("Success").unwrap().into_raw(),

        Err(e) => {
            eprintln!("Failed to retarget filtered GLTF animation: {:?}", e);
            std::ptr::null_mut()
        }
    }
}

pub trait IntoSkeleton {
    fn into_skeleton(self) -> Result<Skeleton, MetaverseMeshAnimationError>;
}

impl IntoSkeleton for &Skeleton {
    fn into_skeleton(self) -> Result<Skeleton, MetaverseMeshAnimationError> {
        Ok(self.clone())
    }
}

impl IntoSkeleton for &Path {
    fn into_skeleton(self) -> Result<Skeleton, MetaverseMeshAnimationError> {
        load_skeleton(self)
    }
}

fn load_skeleton(path: &Path) -> Result<Skeleton, MetaverseMeshAnimationError> {
    let file = fs::File::open(path).map_err(|source| MetaverseMeshAnimationError::ReadFile {
        path: path.to_path_buf(),
        source,
    })?;

    serde_json::from_reader(file).map_err(|source| MetaverseMeshAnimationError::Deserialize {
        path: path.to_path_buf(),
        source,
    })
}

fn used_joints(skeleton: &Skeleton) -> BTreeSet<JointName> {
    skeleton
        .joints
        .values()
        .filter(|joint| {
            joint
                .global_transforms
                .last()
                .is_some_and(|transform| transform.rank != 0)
        })
        .map(|joint| joint.name)
        .collect()
}
