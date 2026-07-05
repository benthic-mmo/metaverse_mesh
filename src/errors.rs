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
