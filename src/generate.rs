use metaverse_messages::utils::render_data::{AvatarObject, RenderObject};
// This file is for generating a mesh that includes a Skeleton object, along with SceneObject
// jsons.
use std::error::Error;
use std::{
    ffi::{CStr, c_char},
    fs,
    path::PathBuf,
};
use crate::gltf::{build_mesh_scene_gltf, build_skinned_mesh_gltf};
pub fn generate_skinned_mesh(
    agent_object: PathBuf,
    out_path: PathBuf,
) -> Result<(), Box<dyn Error>> {
    let json_str =
        fs::read_to_string(&agent_object).expect(&format!("Failed to read {:?}", agent_object));
    let avatar: AvatarObject = serde_json::from_str(&json_str)
        .unwrap_or_else(|e| panic!("Failed to deserialize SceneGroup {:?}", e));
    build_skinned_mesh_gltf(avatar, out_path)    
}
pub fn generate_mesh(
    agent_object: PathBuf,
    out_path: PathBuf,
) -> Result<(), Box<dyn Error>> {
    let json_str =
        fs::read_to_string(&agent_object).expect(&format!("Failed to read {:?}", agent_object));
    let avatar: Vec<RenderObject> = serde_json::from_str(&json_str)
        .unwrap_or_else(|e| panic!("Failed to deserialize SceneGroup {:?}", e));
    build_mesh_scene_gltf(avatar, out_path)?;
    Ok(())
}

#[unsafe(no_mangle)]
/// Allow external projects to generate skinned mesh. This will return the string of where the file
/// was generated on disk!
pub unsafe extern "C" fn generate_skinned_mesh_legacy(
    avatar_object_path: *const *const c_char,
    avatar_object_path_len: usize,
    out_path: *const c_char,
) -> *mut c_char {
    let avatar_object: PathBuf = unsafe {
        std::slice::from_raw_parts(avatar_object_path, avatar_object_path_len)
            .iter()
            .map(|&ptr| {
                let s = CStr::from_ptr(ptr).to_string_lossy().into_owned();
                PathBuf::from(s)
            })
            .collect()
    };

    let out_str = unsafe { CStr::from_ptr(out_path).to_string_lossy().into_owned() };
    let out = PathBuf::from(out_str);
    match generate_skinned_mesh(avatar_object, out) {
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
pub unsafe extern "C" fn generate_mesh_legacy(
    render_object_path: *const *const c_char,
    render_object_path_len: usize,
    out_path: *const c_char,
) -> *mut c_char {
    let render_object: PathBuf = unsafe {
        std::slice::from_raw_parts(render_object_path, render_object_path_len)
            .iter()
            .map(|&ptr| {
                let s = CStr::from_ptr(ptr).to_string_lossy().into_owned();
                PathBuf::from(s)
            })
            .collect()
    };

    let out_str = unsafe { CStr::from_ptr(out_path).to_string_lossy().into_owned() };
    let out = PathBuf::from(out_str);
    match generate_mesh(render_object, out) {
        Ok(_) => std::ffi::CString::new("Success").unwrap().into_raw(),
        Err(e) => {
            eprintln!("Failed to generate mesh: {:?}", e);
            std::ptr::null_mut() // indicate failure to C
        }
    }
}
