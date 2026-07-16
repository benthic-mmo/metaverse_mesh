use crate::animation::gltf::export_filtered_animation;
use benthic_protocol::default_animations::AnimationClip;
use benthic_protocol::skeleton::JointName;
use std::collections::BTreeSet;
use std::error::Error;
use std::str::FromStr;
use std::{
    ffi::{CStr, c_char},
    fs,
    path::PathBuf,
};

pub fn generate_gltf_animation(
    animation_json_path: PathBuf,
    out_path: PathBuf,
    joint_filter: BTreeSet<JointName>,
) -> Result<(), Box<dyn Error>> {
    let json_str = fs::read_to_string(&animation_json_path)
        .expect(&format!("Failed to read {:?}", animation_json_path));
    let animations: AnimationClip = serde_json::from_str(&json_str)
        .unwrap_or_else(|e| panic!("Failed to deserialize joint animation {:?}", e));
    export_filtered_animation(&animations, &joint_filter, out_path)
}

#[unsafe(no_mangle)]
/// Allow external projects to generate animation from json. This will return the string of where the file
/// was generated on disk!
pub unsafe extern "C" fn generate_gltf_animation_legacy(
    animation_json_path: *const c_char,
    out_path: *const c_char,

    // Array of joint name strings
    joint_filter_ptr: *const *const c_char,
    joint_filter_len: usize,
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

    // Parse joint filter
    let joint_filter: BTreeSet<JointName> = unsafe {
        std::slice::from_raw_parts(joint_filter_ptr, joint_filter_len)
            .iter()
            .filter_map(|&ptr| {
                let name = CStr::from_ptr(ptr).to_string_lossy();

                match JointName::from_str(&name) {
                    Ok(joint) => Some(joint),
                    Err(err) => {
                        eprintln!("Invalid joint name {:?}: {:?}", name, err);
                        None
                    }
                }
            })
            .collect()
    };

    // Generate animation
    match generate_gltf_animation(animation_json, out, joint_filter) {
        Ok(_) => std::ffi::CString::new("Success").unwrap().into_raw(),

        Err(e) => {
            eprintln!("Failed to generate joint animation: {:?}", e);
            std::ptr::null_mut()
        }
    }
}
