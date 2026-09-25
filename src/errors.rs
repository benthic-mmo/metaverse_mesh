use std::path::PathBuf;

use benthic_protocol::{errors::SessionError, skeleton::JointName};

#[derive(Debug, thiserror::Error)]
pub enum MetaverseMeshError {
    #[error("boxed error: {0}")]
    BoxedError(#[from] std::boxed::Box<dyn std::error::Error>),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Serde error: {0}")]
    SerdeEerror(#[from] serde_json::Error),

    #[error("Gltf error: {0}")]
    GltfError(#[from] gltf::Error),
}

#[derive(Debug, thiserror::Error)]
pub enum MetaverseMeshAnimationError {
    #[error("joint {joint:?} was not found in the skeleton")]
    MissingJoint { joint: JointName },

    #[error("joint {joint:?} has no {transform_type} transform")]
    MissingTransform {
        joint: JointName,
        transform_type: &'static str,
    },

    #[error("Skeleton contains a cycle at joint {joint}")]
    SkeletonCycle { joint: JointName },

    #[error("Session Error: {0}")]
    SessionError(#[from] SessionError),

    #[error("Failed to read file: {path}, {source}")]
    ReadFile {
        path: PathBuf,
        source: std::io::Error,
    },

    #[error("failed to deserialize animation {path}: {source}")]
    Deserialize {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },

    #[error("failed to create animation output {path}: {source}")]
    CreateOutput {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to serialize animation {path}: {source}")]
    Serialize {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },

    #[error("Mesh Error: {0}")]
    MeshError(#[from] MetaverseMeshError),
}
