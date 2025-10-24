use metaverse_messages::utils::render_data::{AvatarObject, RenderObject};
// This file is for generating a mesh that includes a Skeleton object, along with SceneObject
// jsons.
use metaverse_messages::{http::scene::SceneGroup, utils::skeleton::Skeleton};
use std::error::Error;
use std::{
    ffi::{CStr, c_char},
    fs,
    path::PathBuf,
};

use crate::gltf::{bake_avatar, bake_avatar_2, generate_model};
pub fn generate_skinned_mesh_2(
    agent_object: PathBuf,
    out_path: PathBuf,
) -> Result<(), Box<dyn Error>> {
    let json_str =
        fs::read_to_string(&agent_object).expect(&format!("Failed to read {:?}", agent_object));
    let avatar: AvatarObject = serde_json::from_str(&json_str)
        .unwrap_or_else(|e| panic!("Failed to deserialize SceneGroup {:?}", e));
    bake_avatar_2(avatar, out_path)
}

pub fn generate_skinned_mesh(
    scene_paths: &Vec<PathBuf>,
    skeleton_path: PathBuf,
    out_path: PathBuf,
) -> Result<(), Box<dyn Error>> {
    let json_str =
        fs::read_to_string(&skeleton_path).expect(&format!("Failed to read {:?}", skeleton_path));
    let skeleton: Skeleton = serde_json::from_str(&json_str)
        .unwrap_or_else(|e| panic!("Failed to deserialize Skeleton {:?}", e));

    let mut scenes = Vec::new();
    for scene in scene_paths {
        let json_str = fs::read_to_string(&scene).expect(&format!("Failed to read {:?}", scene));
        let scene: SceneGroup = serde_json::from_str(&json_str)
            .unwrap_or_else(|e| panic!("Failed to deserialize SceneGroup {:?}", e));
        scenes.push(scene);
    }
    bake_avatar(scenes, skeleton, out_path)
}

pub fn generate_mesh(scene_paths: &Vec<PathBuf>, out_path: PathBuf) -> Result<(), Box<dyn Error>> {
    let mut scenes = Vec::new();
    for scene in scene_paths {
        let json_str = fs::read_to_string(&scene).expect(&format!("Failed to read {:?}", scene));
        let scene: SceneGroup = serde_json::from_str(&json_str)
            .unwrap_or_else(|e| panic!("Failed to deserialize SceneGroup {:?}", e));
        scenes.push(scene);
    }
    generate_model(scenes, out_path)
}

#[unsafe(no_mangle)]
/// Allow external projects to generate skinned mesh. This will return the string of where the file
/// was generated on disk!
pub unsafe extern "C" fn generate_skinned_mesh_legacy(
    scene_paths: *const *const c_char,
    scene_paths_len: usize,
    skeleton_path: *const c_char,
    out_path: *const c_char,
) -> *mut c_char {
    let skeleton_str = unsafe { CStr::from_ptr(skeleton_path).to_string_lossy().into_owned() };
    let skeleton = PathBuf::from(skeleton_str);

    let scenes: Vec<PathBuf> = unsafe {
        std::slice::from_raw_parts(scene_paths, scene_paths_len)
            .iter()
            .map(|&ptr| {
                let s = CStr::from_ptr(ptr).to_string_lossy().into_owned();
                PathBuf::from(s)
            })
            .collect()
    };

    let out_str = unsafe { CStr::from_ptr(out_path).to_string_lossy().into_owned() };
    let out = PathBuf::from(out_str);
    match generate_skinned_mesh(&scenes, skeleton, out) {
        Ok(_) => std::ffi::CString::new("Success").unwrap().into_raw(),
        Err(e) => {
            eprintln!("Failed to generate mesh: {:?}", e);
            std::ptr::null_mut() // indicate failure to C
        }
    }
}

#[unsafe(no_mangle)]
/// Allow external projects to generate skinned mesh. This will return the string of where the file
/// was generated on disk!
pub unsafe extern "C" fn generate_model_legacy(
    scene_paths: *const *const c_char,
    scene_paths_len: usize,
    out_path: *const c_char,
) -> *mut c_char {
    let scenes: Vec<PathBuf> = unsafe {
        std::slice::from_raw_parts(scene_paths, scene_paths_len)
            .iter()
            .map(|&ptr| {
                let s = CStr::from_ptr(ptr).to_string_lossy().into_owned();
                PathBuf::from(s)
            })
            .collect()
    };

    let out_str = unsafe { CStr::from_ptr(out_path).to_string_lossy().into_owned() };
    let out = PathBuf::from(out_str);
    match generate_mesh(&scenes, out) {
        Ok(_) => std::ffi::CString::new("Success").unwrap().into_raw(),
        Err(e) => {
            eprintln!("Failed to generate mesh: {:?}", e);
            std::ptr::null_mut() // indicate failure to C
        }
    }
}
