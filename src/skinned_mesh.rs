// This file is for generating a mesh that includes a Skeleton object, along with SceneObject
// jsons.

use std::{ffi::{c_char, CStr}, fs, path::PathBuf};

use metaverse_messages::{capabilities::scene::SceneGroup, utils::skeleton::Skeleton};

use crate::gltf::bake_avatar;

pub fn generate_skinned_mesh(scene_paths: Vec<PathBuf>, skeleton_path: PathBuf, out_path: PathBuf){
    let json_str = fs::read_to_string(&skeleton_path).expect(&format!("Failed to read {:?}", skeleton_path));
    let skeleton: Skeleton = serde_json::from_str(&json_str).unwrap_or_else(|e| panic!("Failed to deserialize Skeleton {:?}", e));
    
    let mut scenes = Vec::new(); 
    for scene in scene_paths{
        let json_str = fs::read_to_string(&scene).expect(&format!("Failed to read {:?}", scene));
        let scene: SceneGroup = serde_json::from_str(&json_str).unwrap_or_else(|e| panic!("Failed to deserialize SceneGroup {:?}", e));
        scenes.push(scene);
    }
    bake_avatar(scenes, skeleton, out_path);
}


#[unsafe(no_mangle)]
/// Allow external projects to generate skinned mesh. This will return the string of where the file
/// was generated on disk! 
pub unsafe extern "C" fn generate_skinned_mesh_legacy(
    scene_paths: *const *const c_char,
    scene_paths_len: usize,

    skeleton_path: *const c_char,
    out_path: *const c_char,
) {
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
    generate_skinned_mesh(scenes, skeleton, out);
}
