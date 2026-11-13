use crate::errors::MetaverseMeshError;
use benthic_protocol::avatar::skeleton::SkinJoint;
use benthic_protocol::avatar::types::{
    AttachmentData, AttachmentTarget, AvatarObject, JointWeight,
};
use benthic_protocol::objects::types::{
    BlinnPhongMaterial, MaterialModel, PBRMaterial, RenderFace, RenderObject,
};
use glam::{Quat, Vec3, usize};
use gltf::material::AlphaMode;
use gltf_json::animation::{Channel, Interpolation, Property, Sampler, Target as ChannelTarget};
use gltf_json::{
    Accessor, Index, Material, Mesh, Node, Scene, Skin, Value,
    accessor::{ComponentType, GenericComponentType},
    buffer::{Stride, Target, View},
    image::MimeType,
    material::{PbrMetallicRoughness, StrengthFactor},
    mesh::{Mode, Primitive, Semantic},
    scene::UnitQuaternion,
    texture,
    validation::{
        Checked::{self, Valid},
        USize64,
    },
};

#[cfg(feature = "ktx2")]
use ktx::{
    ErrorCode, KtxResult, Texture2, TextureCreateInfo, TextureCreateStorage, VkFormat,
    encode_params::{BasisParams, KTX_BASIS_CODEC_UASTC_LDR_4X4, KTX_BASIS_PARAMS_SIZE},
};
use rgb::bytemuck;
#[cfg(feature = "glb")]
use std::borrow::Cow;
use std::collections::HashSet;
use std::f32::consts::FRAC_PI_2;
use std::path::Path;
use std::{
    collections::{BTreeSet, HashMap},
    fs::{self, File},
    path::PathBuf,
    vec,
};
#[cfg(feature = "dxt1")]
use texture_compressor::{AlphaPolicy, Effort, Format, ImageView, Settings};

const KHR_TEXTURE_BASISU: &str = "KHR_texture_basisu";
#[cfg(feature = "dxt1")]
const MSFT_TEXTURE_DDS: &str = "MSFT_texture_dds";

/// Node indices that represent a joint.
#[derive(Debug, Clone)]
pub struct AttachmentNode {
    pub node: Index<Node>,
    pub attachment: AttachmentData,
}

struct GltfBuilder {
    root: gltf_json::Root,
    buffer_index: gltf_json::Index<gltf_json::Buffer>,
    combined_buffer: Vec<u8>,
    nodes: Vec<Index<Node>>,
    mesh_nodes: Vec<Index<Node>>,
    skinned_nodes: Vec<Index<Node>>,
    attachments: Vec<AttachmentNode>,
    ktx2_textures: Vec<(usize, usize)>,
    #[cfg(feature = "dxt1")]
    dds_textures: Vec<(usize, usize)>,
}

impl GltfBuilder {
    // create a new GLTFBuilder, initialized to default values.
    fn new(buffer_name: &str) -> Self {
        let mut root = gltf_json::Root::default();
        let buffer_index = root.push(gltf_json::Buffer {
            byte_length: gltf_json::validation::USize64::from(0_usize),
            name: Some(buffer_name.to_string()),
            uri: None,
            extensions: Default::default(),
            extras: Default::default(),
        });
        Self {
            root,
            buffer_index,
            combined_buffer: Vec::new(),
            nodes: Vec::new(),
            mesh_nodes: Vec::new(),
            skinned_nodes: Vec::new(),
            attachments: Vec::new(),
            ktx2_textures: Vec::new(),
            #[cfg(feature = "dxt1")]
            dds_textures: Vec::new(),
        }
    }

    // GLTFs expect every buffer to be aligned to 4. For some values that are not aligned to 4, this
    // function adds 0 to the end of the buffer which is not used and never read.
    fn align_4(&mut self) {
        while !self.combined_buffer.len().is_multiple_of(4) {
            self.combined_buffer.push(0);
        }
    }

    fn push_view(
        &mut self,
        byte_length: usize,
        byte_stride: Option<usize>,
        target: Option<gltf_json::validation::Checked<gltf_json::buffer::Target>>,
        name: String,
    ) -> gltf_json::Index<gltf_json::buffer::View> {
        let offset = self.combined_buffer.len();
        self.root.push(View {
            buffer: self.buffer_index,
            byte_length: USize64::from(byte_length),
            byte_offset: Some(USize64::from(offset)),
            byte_stride: byte_stride.map(Stride),
            target,
            extensions: Default::default(),
            extras: Default::default(),
            name: Some(name.to_string()),
        })
    }

    fn add_vertex_positions(&mut self, vertices: &[Vec3]) -> gltf_json::Index<gltf_json::Accessor> {
        let (min, max) = bounding_coords(vertices);
        let vertex_bytes = to_padded_byte_vector(vertices);
        self.align_4();

        let view = self.push_view(
            vertex_bytes.len(),
            Some(std::mem::size_of::<Vec3>()),
            Some(Valid(Target::ArrayBuffer)),
            "vertex_positions".to_string(),
        );
        self.combined_buffer.extend_from_slice(&vertex_bytes);

        self.root.push(gltf_json::Accessor {
            buffer_view: Some(view),
            byte_offset: Some(USize64(0)),
            count: USize64::from(vertices.len()),
            component_type: Valid(GenericComponentType(ComponentType::F32)),
            type_: Valid(gltf_json::accessor::Type::Vec3),
            min: Some(Value::from(Vec::from(min))),
            max: Some(Value::from(Vec::from(max))),
            normalized: false,
            sparse: None,
            name: Some("POSITION".to_string()),
            extensions: Default::default(),
            extras: Default::default(),
        })
    }

    fn add_normals(&mut self, normals: &[Vec3]) -> Index<Accessor> {
        let bytes: Vec<u8> =
            bytemuck::cast_slice(&normals.iter().map(|n| [n.x, n.y, n.z]).collect::<Vec<_>>())
                .to_vec();

        self.align_4();

        let view = self.push_view(
            bytes.len(),
            Some(std::mem::size_of::<[f32; 3]>()),
            Some(Valid(Target::ArrayBuffer)),
            "normals".to_string(),
        );

        self.combined_buffer.extend_from_slice(&bytes);

        self.root.push(gltf_json::Accessor {
            buffer_view: Some(view),
            byte_offset: Some(USize64(0)),
            count: USize64::from(normals.len()),
            component_type: Valid(GenericComponentType(ComponentType::F32)),
            type_: Valid(gltf_json::accessor::Type::Vec3),
            min: None,
            max: None,
            normalized: false,
            sparse: None,
            name: Some("NORMAL".to_string()),
            extensions: Default::default(),
            extras: Default::default(),
        })
    }

    fn add_indices(&mut self, indices: &[u32]) -> gltf_json::Index<gltf_json::Accessor> {
        let mut bytes = Vec::with_capacity(indices.len() * std::mem::size_of::<u32>());

        for index in indices {
            bytes.extend_from_slice(&index.to_le_bytes());
        }

        self.align_4();

        let view = self.push_view(
            bytes.len(),
            None,
            Some(Valid(Target::ElementArrayBuffer)),
            "indices".to_string(),
        );

        self.combined_buffer.extend_from_slice(&bytes);

        self.root.push(gltf_json::Accessor {
            buffer_view: Some(view),
            byte_offset: Some(USize64(0)),
            count: USize64::from(indices.len()),
            component_type: Valid(GenericComponentType(ComponentType::U32)),
            type_: Valid(gltf_json::accessor::Type::Scalar),
            normalized: false,
            sparse: None,
            name: Some("INDICES".to_string()),
            extensions: Default::default(),
            extras: Default::default(),
            min: None,
            max: None,
        })
    }

    // This generates a single animation frame of the bind pose and applies it to the skeleton.
    // This allows some engines to register that this model should be animated.
    pub fn add_bind_pose_animation(
        &mut self,
        avatar: &AvatarObject,
        bones: &BTreeSet<SkinJoint>,
        joint_to_node: &HashMap<SkinJoint, Index<Node>>,
        effective_parents: &HashMap<SkinJoint, Option<SkinJoint>>,
    ) {
        use gltf_json::*;

        let mut input_bytes = Vec::new();
        input_bytes.extend_from_slice(&0.0f32.to_le_bytes());

        self.align_4();
        let input_offset = self.combined_buffer.len();
        self.combined_buffer.extend_from_slice(&input_bytes);

        let input_view = self.root.push(View {
            buffer: self.buffer_index,
            byte_length: input_bytes.len().into(),
            byte_offset: Some(USize64(input_offset as u64)),
            byte_stride: None,
            target: None,
            name: Some("animation_input".to_string()),
            extensions: Default::default(),
            extras: Default::default(),
        });

        let input_accessor = self.root.push(Accessor {
            buffer_view: Some(input_view),
            byte_offset: Some(USize64(0)),
            count: USize64(1),
            component_type: Checked::Valid(GenericComponentType(ComponentType::F32)),
            type_: Checked::Valid(accessor::Type::Scalar),
            normalized: false,
            min: Some(Value::from(vec![0.0])),
            max: Some(Value::from(vec![0.0])),
            sparse: None,
            name: Some("time_0".to_string()),
            extensions: Default::default(),
            extras: Default::default(),
        });

        let mut channels = Vec::new();
        let mut samplers = Vec::new();

        for joint_name in bones {
            let node_index = joint_to_node[joint_name];
            let joint = &avatar.global_skeleton.joints[joint_name];

            let global_transform = joint.global_transforms.last().unwrap().transform;

            let local_transform = match effective_parents[joint_name] {
                Some(parent_name) => {
                    let parent_global = avatar
                        .global_skeleton
                        .joints
                        .get(&parent_name)
                        .unwrap()
                        .global_transforms
                        .last()
                        .unwrap()
                        .transform;

                    parent_global.inverse() * global_transform
                }
                None => global_transform,
            };

            let (scale, rotation, translation) = local_transform.to_scale_rotation_translation();

            let push_accessor_vec3 =
                |builder: &mut GltfBuilder, vec: Vec3, name: &str| -> Index<Accessor> {
                    let bytes: Vec<u8> = bytemuck::cast_slice(&[[vec.x, vec.y, vec.z]]).to_vec();

                    builder.align_4();
                    let offset = builder.combined_buffer.len();
                    builder.combined_buffer.extend_from_slice(&bytes);

                    let view = builder.root.push(View {
                        buffer: builder.buffer_index,
                        byte_length: bytes.len().into(),
                        byte_offset: Some(USize64(offset as u64)),
                        byte_stride: None,
                        target: None,
                        name: Some(name.to_string()),
                        extensions: Default::default(),
                        extras: Default::default(),
                    });

                    builder.root.push(Accessor {
                        buffer_view: Some(view),
                        byte_offset: Some(USize64(0)),
                        count: USize64(1),
                        component_type: Checked::Valid(GenericComponentType(ComponentType::F32)),
                        type_: Checked::Valid(accessor::Type::Vec3),
                        normalized: false,
                        min: Some(Value::from([vec.x, vec.y, vec.z])),
                        max: Some(Value::from([vec.x, vec.y, vec.z])),
                        sparse: None,
                        name: Some(name.to_string()),
                        extensions: Default::default(),
                        extras: Default::default(),
                    })
                };

            let push_accessor_quat =
                |builder: &mut GltfBuilder, q: Quat, name: &str| -> Index<Accessor> {
                    let bytes: Vec<u8> = bytemuck::cast_slice(&[[q.x, q.y, q.z, q.w]]).to_vec();

                    builder.align_4();
                    let offset = builder.combined_buffer.len();
                    builder.combined_buffer.extend_from_slice(&bytes);

                    let view = builder.root.push(View {
                        buffer: builder.buffer_index,
                        byte_length: bytes.len().into(),
                        byte_offset: Some(USize64(offset as u64)),
                        byte_stride: None,
                        target: None,
                        name: Some(name.to_string()),
                        extensions: Default::default(),
                        extras: Default::default(),
                    });

                    builder.root.push(Accessor {
                        buffer_view: Some(view),
                        byte_offset: Some(USize64(0)),
                        count: USize64(1),
                        component_type: Checked::Valid(GenericComponentType(ComponentType::F32)),
                        type_: Checked::Valid(accessor::Type::Vec4),
                        normalized: false,
                        min: Some(Value::from([q.x, q.y, q.z, q.w])),
                        max: Some(Value::from([q.x, q.y, q.z, q.w])),
                        sparse: None,
                        name: Some(name.to_string()),
                        extensions: Default::default(),
                        extras: Default::default(),
                    })
                };

            let t_acc = push_accessor_vec3(self, translation, &format!("{}_T", joint_name));
            let r_acc = push_accessor_quat(self, rotation, &format!("{}_R", joint_name));
            let s_acc = push_accessor_vec3(self, scale, &format!("{}_S", joint_name));

            for (path_str, acc) in &[
                ("translation", t_acc),
                ("rotation", r_acc),
                ("scale", s_acc),
            ] {
                let sampler_index = samplers.len();

                samplers.push(Sampler {
                    input: input_accessor,
                    interpolation: Valid(Interpolation::Step),
                    output: *acc,
                    extensions: Default::default(),
                    extras: Default::default(),
                });

                channels.push(Channel {
                    sampler: Index::new(sampler_index as u32),
                    target: ChannelTarget {
                        node: node_index,
                        path: match *path_str {
                            "translation" => Valid(Property::Translation),
                            "rotation" => Valid(Property::Rotation),
                            "scale" => Valid(Property::Scale),
                            _ => panic!("invalid path"),
                        },
                        extensions: Default::default(),
                        extras: Default::default(),
                    },
                    extensions: Default::default(),
                    extras: Default::default(),
                });
            }
        }

        self.root.push(Animation {
            name: Some("BindPose".to_string()),
            channels,
            samplers,
            extensions: Default::default(),
            extras: Default::default(),
        });
    }

    fn add_joint_data(
        &mut self,
        skin_weights: Vec<JointWeight>,
        bones: &BTreeSet<SkinJoint>,
    ) -> (Option<Index<Accessor>>, Option<Index<Accessor>>) {
        if skin_weights.is_empty() {
            return (None, None);
        }

        let bone_index: HashMap<SkinJoint, u8> = bones
            .iter()
            .enumerate()
            .map(|(i, j)| (*j, i as u8))
            .collect();

        let mut joint_indices_bytes = Vec::new();
        let mut joint_weights_bytes = Vec::new();

        for vw in &skin_weights {
            let mut joints = [0u8; 4];
            let mut weights = [0.0f32; 4];

            for i in 0..4 {
                if let (Some(skin_joint), Some(&weight)) = (vw.joint_name.get(i), vw.weights.get(i))
                    && weight > 0.0
                    && let Some(&idx) = bone_index.get(skin_joint)
                {
                    joints[i] = idx;
                    weights[i] = weight;
                }
            }

            let sum: f32 = weights.iter().sum();
            if sum > 0.0 {
                for w in &mut weights {
                    *w /= sum;
                }
            }

            joint_indices_bytes.extend_from_slice(&joints);
            for w in &weights {
                joint_weights_bytes.extend_from_slice(&w.to_le_bytes());
            }
        }

        self.align_4();
        let indices_offset = self.combined_buffer.len();
        self.combined_buffer.extend_from_slice(&joint_indices_bytes);

        let indices_view = self.root.push(View {
            buffer: self.buffer_index,
            byte_length: USize64::from(joint_indices_bytes.len()),
            byte_offset: Some(USize64::from(indices_offset)),
            byte_stride: Some(Stride(4)),
            target: Some(Checked::Valid(Target::ArrayBuffer)),
            extensions: None,
            extras: Default::default(),
            name: Some("joint_indices".into()),
        });

        let indices_accessor = self.root.push(Accessor {
            buffer_view: Some(indices_view),
            byte_offset: Some(USize64(0)),
            count: USize64::from(skin_weights.len()),
            component_type: Checked::Valid(GenericComponentType(ComponentType::U8)),
            type_: Checked::Valid(gltf_json::accessor::Type::Vec4),
            normalized: false,
            sparse: None,
            extensions: None,
            extras: Default::default(),
            name: Some("JOINTS_0".into()),
            min: None,
            max: None,
        });

        self.align_4();
        let weights_offset = self.combined_buffer.len();
        self.combined_buffer.extend_from_slice(&joint_weights_bytes);

        let weights_view = self.root.push(View {
            buffer: self.buffer_index,
            byte_length: USize64::from(joint_weights_bytes.len()),
            byte_offset: Some(USize64::from(weights_offset)),
            byte_stride: Some(Stride(16)),
            target: Some(Checked::Valid(Target::ArrayBuffer)),
            extensions: None,
            extras: Default::default(),
            name: Some("joint_weights".into()),
        });

        let weights_accessor = self.root.push(Accessor {
            buffer_view: Some(weights_view),
            byte_offset: Some(USize64(0)),
            count: USize64::from(skin_weights.len()),
            component_type: Checked::Valid(GenericComponentType(ComponentType::F32)),
            type_: Checked::Valid(gltf_json::accessor::Type::Vec4),
            normalized: false,
            sparse: None,
            extensions: None,
            extras: Default::default(),
            name: Some("WEIGHTS_0".into()),
            min: None,
            max: None,
        });

        (Some(indices_accessor), Some(weights_accessor))
    }

    pub fn add_inverse_bind_matrices(&mut self, ibm_matrices: &[[f32; 16]]) -> Index<Accessor> {
        let offset = self.combined_buffer.len();
        for mat in ibm_matrices {
            for f in mat.iter() {
                self.combined_buffer.extend_from_slice(&f.to_le_bytes());
            }
        }

        let view = self.root.push(View {
            buffer: self.buffer_index,
            byte_length: USize64::from(ibm_matrices.len() * 16 * std::mem::size_of::<f32>()),
            byte_offset: Some(USize64::from(offset)),
            byte_stride: None,
            target: None,
            name: Some("inverse_bind_matrices_view".to_string()),
            extensions: Default::default(),
            extras: Default::default(),
        });

        self.root.push(Accessor {
            buffer_view: Some(view),
            byte_offset: Some(USize64(0)),
            count: USize64::from(ibm_matrices.len()),
            component_type: Checked::Valid(GenericComponentType(ComponentType::F32)),
            type_: Checked::Valid(gltf_json::accessor::Type::Mat4),
            normalized: false,
            sparse: None,
            extensions: None,
            extras: Default::default(),
            name: Some("inverse_bind_matrices_accessor".to_string()),
            min: None,
            max: None,
        })
    }

    pub fn add_mesh(
        &mut self,
        object: RenderObject,
        used_joints: Option<&BTreeSet<SkinJoint>>,
    ) -> gltf_json::Index<gltf_json::Mesh> {
        let mut primitives = Vec::new();

        for face in &object.faces {
            if face.vertices.is_empty() || face.indices.is_empty() {
                continue;
            }

            let pos_accessor = self.add_vertex_positions(&face.vertices);

            let mut attributes: std::collections::BTreeMap<_, _> =
                [(Valid(Semantic::Positions), pos_accessor)]
                    .into_iter()
                    .collect();

            // Normals are per-face vertex attributes.
            if let Some(normals) = &face.normals {
                if normals.len() == face.vertices.len() {
                    let normal_accessor = self.add_normals(normals);
                    attributes.insert(Valid(Semantic::Normals), normal_accessor);
                }
            }

            // UVs and texture material belong to this face's primitive.
            let material = if face.uv.len() == face.vertices.len() {
                let uv_accessor = self.add_uvs(&face.uv);
                attributes.insert(Valid(Semantic::TexCoords(0)), uv_accessor);

                if let Some(material) = &face.material {
                    match material {
                        MaterialModel::BlinnPhong(blinn_phong_material) => {
                            self.add_blinn_phong_texture(face, blinn_phong_material)
                        }
                        MaterialModel::PBR(pbr_material) => {
                            self.add_pbr_texture(face, pbr_material)
                        }
                    }
                } else {
                    None
                }
            } else {
                None
            };

            // Each face has its own local index buffer, so do not offset its indices.
            let index_accessor = self.add_indices(&face.indices);

            // Skin attributes must correspond to this face's vertices.
            if object.skin.is_some()
                && let Some(used_joints) = used_joints
            {
                if let Some(weights) = &face.weights {
                    if weights.len() == face.vertices.len() {
                        let (joint_indices, joint_weights) =
                            self.add_joint_data(weights.clone(), used_joints);

                        if let Some(joints) = joint_indices {
                            attributes.insert(Valid(Semantic::Joints(0)), joints);
                        }

                        if let Some(weights) = joint_weights {
                            attributes.insert(Valid(Semantic::Weights(0)), weights);
                        }
                    }
                }
            }

            primitives.push(Primitive {
                attributes,
                indices: Some(index_accessor),
                material,
                mode: Valid(Mode::Triangles),
                targets: None,
                extensions: Default::default(),
                extras: Default::default(),
            });
        }

        let mesh_index = self.root.push(Mesh {
            primitives,
            weights: None,
            extensions: Default::default(),
            extras: Default::default(),
            name: Some(object.name.clone()),
        });

        // Create one scene node for the whole GLTF mesh.
        let node_index = self.add_node_with_mesh(mesh_index, &object.name);
        // also handle attachment data
        if let Some(attach_data) = object.attachment_data {
            self.attachments.push(AttachmentNode {
                node: node_index,
                attachment: attach_data,
            })
        }

        self.mesh_nodes.push(node_index);

        if object.skin.is_some() {
            self.skinned_nodes.push(node_index);
        }

        mesh_index
    }

    fn add_pbr_texture(
        &self,
        _face: &RenderFace,
        _pbr: &PBRMaterial,
    ) -> Option<gltf_json::Index<gltf_json::Material>> {
        None
    }

    fn add_blinn_phong_texture(
        &mut self,
        _face: &RenderFace,
        blinn_phong: &BlinnPhongMaterial,
    ) -> Option<Index<gltf_json::Material>> {
        let extension = blinn_phong
            .texture_path
            .extension()
            .and_then(|s| s.to_str())?;

        let mime_type = match extension.to_ascii_lowercase().as_str() {
            "png" => "image/png",
            "jpg" | "jpeg" => "image/jpeg",
            _ => return None,
        };

        let is_png = extension.eq_ignore_ascii_case("png");

        // Always register the original image as the standard glTF fallback.
        let image_index = self.add_image(&blinn_phong.texture_path, "diffuse", mime_type)?;

        let texture_index = self.root.push(gltf_json::Texture {
            sampler: None,
            source: image_index,
            name: Some("diffuse_texture".to_string()),
            extensions: Default::default(),
            extras: Default::default(),
        });

        #[cfg(feature = "dxt1")]
        let mut using_dxt1 = false;

        // Register optional BC1 DDS texture.
        #[cfg(feature = "dxt1")]
        if is_png {
            let dds_path = dxt1_dds_path(&blinn_phong.texture_path);

            if let Err(e) = png_to_dxt1_dds(&blinn_phong.texture_path, &dds_path) {
                eprintln!(
                    "Failed to convert PNG to BC1 DDS (using PNG fallback): {}: {:?}",
                    blinn_phong.texture_path.display(),
                    e
                );
            }

            if dds_path.is_file() {
                if let Some(dds_image_index) =
                    self.add_image(&dds_path, "diffuse_dds", "image/vnd-ms.dds")
                {
                    self.dds_textures
                        .push((texture_index.value(), dds_image_index.value()));

                    using_dxt1 = true;
                }
            }
        }

        // Preserve optional Basis Universal KTX2 support when DXT1 is not selected.
        #[cfg(feature = "ktx2")]
        {
            #[cfg(feature = "dxt1")]
            let should_add_ktx2 = !using_dxt1;

            #[cfg(not(feature = "dxt1"))]
            let should_add_ktx2 = true;

            if should_add_ktx2 {
                let ktx2_path = blinn_phong
                    .ktx2_path
                    .clone()
                    .or_else(|| is_png.then(|| blinn_phong.texture_path.with_extension("ktx2")));

                if let Some(ktx2_path) = ktx2_path {
                    if !ktx2_path.is_file() && is_png {
                        match png_to_ktx2(&blinn_phong.texture_path, &ktx2_path) {
                            Ok(()) => {
                                println!("Generated KTX2 texture: {}", ktx2_path.display());
                            }
                            Err(e) => {
                                eprintln!(
                                    "Failed to convert PNG to Basis UASTC KTX2: {}: {:?}",
                                    blinn_phong.texture_path.display(),
                                    e
                                );
                            }
                        }
                    }

                    if ktx2_path.is_file() {
                        if let Some(ktx2_image_index) =
                            self.add_image(&ktx2_path, "diffuse_ktx2", "image/ktx2")
                        {
                            self.ktx2_textures
                                .push((texture_index.value(), ktx2_image_index.value()));
                        }
                    }
                }
            }
        }

        // Determine alpha mode using the original image.
        let has_alpha = image::open(&blinn_phong.texture_path)
            .map(|img| img.to_rgba8().pixels().any(|p| p.0[3] < 255))
            .unwrap_or(false);

        let alpha_mode = if blinn_phong.color.a < 255 || has_alpha {
            Checked::Valid(AlphaMode::Mask)
        } else {
            Checked::Valid(AlphaMode::Opaque)
        };

        // Texture transform.
        let transform = &blinn_phong.texture_transform;

        let texture_info = texture::Info {
            index: texture_index,
            tex_coord: 0,
            extensions: Some(gltf_json::extensions::texture::Info {
                texture_transform: Some(gltf_json::extensions::texture::TextureTransform {
                    offset: gltf_json::extensions::texture::TextureTransformOffset([
                        transform.offset_u,
                        1.0 - transform.offset_v,
                    ]),
                    rotation: gltf_json::extensions::texture::TextureTransformRotation(
                        -transform.rotation,
                    ),
                    scale: gltf_json::extensions::texture::TextureTransformScale([
                        transform.repeat_u,
                        transform.repeat_v,
                    ]),
                    tex_coord: Some(0),
                    extras: Default::default(),
                }),
                others: Default::default(),
            }),
            extras: Default::default(),
        };

        let material_index = self.root.push(Material {
            pbr_metallic_roughness: PbrMetallicRoughness {
                base_color_texture: Some(texture_info),
                base_color_factor: gltf_json::material::PbrBaseColorFactor([
                    blinn_phong.color.r as f32 / 255.0,
                    blinn_phong.color.g as f32 / 255.0,
                    blinn_phong.color.b as f32 / 255.0,
                    blinn_phong.color.a as f32 / 255.0,
                ]),
                metallic_factor: StrengthFactor(0.0),
                roughness_factor: StrengthFactor(1.0),
                ..Default::default()
            },
            alpha_mode,
            name: Some("material_with_texture".to_string()),
            ..Default::default()
        });

        Some(material_index)
    }
    fn add_image(
        &mut self,
        path: &Path,
        name: &str,
        mime_type: &str,
    ) -> Option<Index<gltf_json::Image>> {
        #[cfg(feature = "glb")]
        {
            let image_data = fs::read(path).ok()?;
            self.add_image_buffer(&image_data, name, mime_type)
        }

        #[cfg(not(feature = "glb"))]
        {
            // Reference the original image path directly.
            // Do not copy or rename the image.
            let uri = path.to_string_lossy().replace('\\', "/");

            Some(self.root.push(gltf_json::Image {
                uri: Some(uri),
                mime_type: None,
                buffer_view: None,
                name: Some(name.to_string()),
                extensions: Default::default(),
                extras: Default::default(),
            }))
        }
    }

    #[cfg(feature = "glb")]
    fn add_image_buffer(
        &mut self,
        image_data: &[u8],
        name: &str,
        mime_type: &str,
    ) -> Option<Index<gltf_json::Image>> {
        self.align_4();

        let offset = self.combined_buffer.len();
        self.combined_buffer.extend_from_slice(image_data);

        let buffer_view = self.root.push(View {
            buffer: self.buffer_index,
            byte_length: image_data.len().into(),
            byte_offset: Some(USize64::from(offset)),
            byte_stride: None,
            target: None,
            name: Some(format!("{name}_buffer_view")),
            extensions: Default::default(),
            extras: Default::default(),
        });

        Some(self.root.push(gltf_json::Image {
            uri: None,
            mime_type: Some(MimeType(mime_type.to_string())),
            buffer_view: Some(buffer_view),
            name: Some(name.to_string()),
            extensions: Default::default(),
            extras: Default::default(),
        }))
    }

    #[cfg(feature = "dxt1")]
    pub fn png_to_dxt1_dds(input_path: &Path, output_path: &Path) -> std::io::Result<()> {
        use std::io::Write;

        let rgba = image::open(input_path)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?
            .to_rgba8();

        let (width, height) = rgba.dimensions();

        if width == 0 || height == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid texture dimensions",
            ));
        }

        let has_alpha = rgba.pixels().any(|pixel| pixel.0[3] < 255);
        let pixels = rgba.into_raw();

        let image_view = ImageView::rgba8(&pixels, width, height, width as usize * 4)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("{e:?}")))?;

        let settings = Settings::new(Format::Bc1, Effort::Balanced).with_alpha(if has_alpha {
            AlphaPolicy::PunchThrough { threshold: 128 }
        } else {
            AlphaPolicy::Opaque
        });

        let compressed = texture_compressor::encode(image_view, &settings)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("{e:?}")))?;

        let mut file = File::create(output_path)?;

        // DDS magic.
        file.write_all(b"DDS ")?;

        // DDS_HEADER.
        let flags = 0x0008_1007u32; // CAPS | HEIGHT | WIDTH | PIXELFORMAT | LINEARSIZE

        for value in [
            124,                     // dwSize
            flags,                   // dwFlags
            height,                  // dwHeight
            width,                   // dwWidth
            compressed.len() as u32, // dwPitchOrLinearSize
            0,                       // dwDepth
            0,                       // dwMipMapCount
        ] {
            file.write_all(&value.to_le_bytes())?;
        }

        // dwReserved1[11].
        for _ in 0..11 {
            file.write_all(&0u32.to_le_bytes())?;
        }

        // DDS_PIXELFORMAT.
        for value in [
            32,                           // dwSize
            0x4,                          // DDPF_FOURCC
            u32::from_le_bytes(*b"DX10"), // dwFourCC
            0,                            // dwRGBBitCount
            0,                            // dwRBitMask
            0,                            // dwGBitMask
            0,                            // dwBBitMask
            0,                            // dwABitMask
        ] {
            file.write_all(&value.to_le_bytes())?;
        }

        // DDS_HEADER caps.
        for value in [
            0x1000u32, // DDSCAPS_TEXTURE
            0,         // dwCaps2
            0,         // dwCaps3
            0,         // dwCaps4
            0,         // dwReserved2
        ] {
            file.write_all(&value.to_le_bytes())?;
        }

        // DDS_HEADER_DXT10.
        for value in [
            72u32, // DXGI_FORMAT_BC1_UNORM_SRGB
            3,     // D3D10_RESOURCE_DIMENSION_TEXTURE2D
            0,     // miscFlag
            1,     // arraySize
            0,     // miscFlags2
        ] {
            file.write_all(&value.to_le_bytes())?;
        }

        // BC1 compressed image data.
        file.write_all(&compressed)?;

        Ok(())
    }

    #[cfg(feature = "dxt1")]
    fn apply_dxt1_extensions(&self, json: &mut serde_json::Value) {
        if self.dds_textures.is_empty() {
            return;
        }

        let textures = json["textures"]
            .as_array_mut()
            .expect("Missing textures array");

        for (texture_index, dds_image_index) in &self.dds_textures {
            textures[*texture_index]["extensions"][MSFT_TEXTURE_DDS] = serde_json::json!({
                "source": dds_image_index
            });
        }
    }

    fn apply_ktx2_extensions(&self, json: &mut serde_json::Value) {
        if self.ktx2_textures.is_empty() {
            return;
        }

        let textures = json["textures"]
            .as_array_mut()
            .expect("Missing textures array");

        for (texture_index, ktx2_image_index) in &self.ktx2_textures {
            let texture = &mut textures[*texture_index];

            texture["extensions"][KHR_TEXTURE_BASISU] = serde_json::json!({
                "source": ktx2_image_index
            });
        }
    }
    pub fn add_node_with_mesh(
        &mut self,
        mesh_index: gltf_json::Index<gltf_json::Mesh>,
        name: &str,
    ) -> Index<Node> {
        let node_index = self.root.push(Node {
            mesh: Some(mesh_index),
            name: Some(name.to_string()),
            ..Default::default()
        });
        self.nodes.push(node_index);
        node_index
    }
    pub fn add_uvs(&mut self, uvs: &[[f32; 2]]) -> Index<Accessor> {
        let bytes: Vec<u8> = bytemuck::cast_slice(uvs).to_vec();
        self.align_4();

        let view = self.push_view(
            bytes.len(),
            Some(std::mem::size_of::<[f32; 2]>()),
            Some(Checked::Valid(Target::ArrayBuffer)),
            "uvs".to_string(),
        );
        self.combined_buffer.extend_from_slice(&bytes);

        self.root.push(gltf_json::Accessor {
            buffer_view: Some(view),
            byte_offset: Some(USize64(0)),
            count: USize64::from(uvs.len()),
            component_type: Checked::Valid(GenericComponentType(ComponentType::F32)),
            type_: Checked::Valid(gltf_json::accessor::Type::Vec2),
            min: None,
            max: None,
            normalized: false,
            sparse: None,
            name: Some("TEXCOORD_0".to_string()),
            extensions: Default::default(),
            extras: Default::default(),
        })
    }

    pub fn finalize_scene(&mut self, name: &str) {
        // Root node containing all scene nodes
        let root_node_index = self.root.push(Node {
            children: Some(self.nodes.clone()),
            name: Some(format!("{name}_root")),
            ..Default::default()
        });

        // Push the scene referencing that root node
        self.root.push(Scene {
            extensions: Default::default(),
            extras: Default::default(),
            name: Some(name.to_string()),
            nodes: vec![root_node_index],
        });

        // clear for potential next scene
        self.nodes.clear();
    }

    pub fn rotated_finalize_scene(&mut self, name: &str) {
        let rotation = Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2);

        // Wrap existing nodes under this rotated root
        let rotated_root_index = self.root.push(Node {
            children: Some(self.nodes.clone()),
            name: Some("RotatedRoot".to_string()),
            rotation: Some(UnitQuaternion([
                rotation.x, rotation.y, rotation.z, rotation.w,
            ])),
            ..Default::default()
        });

        // Create scene referencing rotated root
        self.root.push(Scene {
            nodes: vec![rotated_root_index],
            name: Some(format!("{}_rotated_scene", name)),
            extensions: Default::default(),
            extras: Default::default(),
        });

        self.nodes.clear();
    }

    #[cfg(feature = "glb")]
    fn finalize(mut self, path: &Path) -> Result<PathBuf, Box<dyn std::error::Error>> {
        self.prepare_root();
        self.write_glb(path)
    }

    #[cfg(not(feature = "glb"))]
    fn finalize(mut self, path: &Path) -> Result<PathBuf, Box<dyn std::error::Error>> {
        self.prepare_root();
        self.write_gltf_bin(path)
    }

    fn prepare_root(&mut self) {
        self.root.buffers[self.buffer_index.value()].byte_length =
            USize64::from(self.combined_buffer.len());

        self.root
            .extensions_used
            .push("KHR_texture_transform".to_string());

        if !self.ktx2_textures.is_empty()
            && !self
                .root
                .extensions_used
                .iter()
                .any(|e| e == KHR_TEXTURE_BASISU)
        {
            self.root
                .extensions_used
                .push(KHR_TEXTURE_BASISU.to_string());
        }

        #[cfg(feature = "dxt1")]
        if !self.dds_textures.is_empty()
            && !self
                .root
                .extensions_used
                .iter()
                .any(|e| e == MSFT_TEXTURE_DDS)
        {
            self.root.extensions_used.push(MSFT_TEXTURE_DDS.to_string());
        }
    }

    #[cfg(feature = "glb")]
    fn write_glb(self, path: &Path) -> Result<PathBuf, Box<dyn std::error::Error>> {
        let json = self.root_to_json()?;
        let json_string = serde_json::to_string(&json)?;

        let glb = gltf::binary::Glb {
            header: gltf::binary::Header {
                magic: *b"glTF",
                version: 2,
                length: (json_string.len() + self.combined_buffer.len()).try_into()?,
            },
            json: Cow::Owned(json_string.into_bytes()),
            bin: Some(Cow::Owned(self.combined_buffer)),
        };
        let file = File::create(path).map_err(|e| {
        format!("File::create failed:\n  path: {:?}\n  exists: {}\n  parent: {:?}\n  parent_exists: {}\n  error: {:?}",
            path, path.exists(), path.parent(),
            path.parent().map(|p| p.exists()).unwrap_or(false), e)
    })?;

        glb.to_writer(file)?;
        Ok(path.to_path_buf())
    }

    #[cfg(not(feature = "glb"))]
    fn write_gltf_bin(self, path: &Path) -> Result<PathBuf, Box<dyn std::error::Error>> {
        let bin_path = path.with_extension("bin");
        let bin_name = bin_path
            .file_name()
            .and_then(|s| s.to_str())
            .ok_or("invalid bin file name")?
            .to_string();

        let mut json = self.root_to_json()?;

        // Reference the external buffer file from the glTF JSON.
        json["buffers"][self.buffer_index.value()]["uri"] = serde_json::Value::String(bin_name);

        let json_string = serde_json::to_string_pretty(&json)?;

        fs::write(&bin_path, &self.combined_buffer)?;
        fs::write(path, json_string)?;

        Ok(path.to_path_buf())
    }

    fn root_to_json(&self) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        let mut json: serde_json::Value =
            serde_json::from_str(&gltf_json::serialize::to_string(&self.root)?)?;

        #[cfg(feature = "ktx2")]
        self.apply_ktx2_extensions(&mut json);

        #[cfg(feature = "dxt1")]
        self.apply_dxt1_extensions(&mut json);

        Ok(json)
    }

    pub fn finalize_avatar(&mut self, avatar: &AvatarObject) {
        let bones = &avatar.used_joints;

        if bones.is_empty() {
            self.finalize_scene("AvatarScene");
            return;
        }

        let mut joint_to_node: HashMap<SkinJoint, Index<Node>> = HashMap::new();

        let mut skeleton_nodes = Vec::new();
        let mut ibm_matrices = Vec::new();
        let effective_parents = find_effective_parents(avatar);

        // Create joint nodes.
        for joint_name in bones {
            // if the joint isn't in the global skeleton, don't create it
            let Some(joint) = avatar.global_skeleton.joints.get(joint_name) else {
                continue;
            };

            let global_transform = joint.global_transforms.last().unwrap().transform;

            let local_transform = match effective_parents[joint_name] {
                Some(parent_name) => {
                    let parent_global = avatar
                        .global_skeleton
                        .joints
                        .get(&parent_name)
                        .unwrap()
                        .global_transforms
                        .last()
                        .unwrap()
                        .transform;

                    parent_global.inverse() * global_transform
                }
                None => global_transform,
            };

            let (scale, rotation, translation) = local_transform.to_scale_rotation_translation();

            let joint_node_index = self.root.push(Node {
                name: Some(joint_name.to_string()),
                scale: Some(scale.into()),
                rotation: Some(UnitQuaternion([
                    rotation.x, rotation.y, rotation.z, rotation.w,
                ])),
                translation: Some(translation.into()),
                ..Default::default()
            });

            joint_to_node.insert(*joint_name, joint_node_index);
            skeleton_nodes.push(joint_node_index);

            ibm_matrices.push(global_transform.inverse().to_cols_array());
        }

        // Connect joints to their nearest used ancestor.
        for joint_name in bones {
            let child_index = joint_to_node[joint_name];

            if let Some(Some(parent_name)) = effective_parents.get(joint_name) {
                let parent_index = joint_to_node[parent_name];

                self.root.nodes[parent_index.value()]
                    .children
                    .get_or_insert_with(Vec::new)
                    .push(child_index);
            }
        }

        // Attach rigid meshes to their target joints.
        for attachment_node in &self.attachments {
            let attachment = &attachment_node.attachment;

            let target: AttachmentTarget = attachment.attach_point.into();

            let joint = match target {
                AttachmentTarget::Bone(name) => SkinJoint::Joint(name),
                AttachmentTarget::DeformJoint(name) => SkinJoint::Deform(name),
                AttachmentTarget::HUDAttachment(_) => {
                    // HUD attachments should not be handled by the GLTF builder.
                    continue;
                }
            };

            let Some(&joint_node) = joint_to_node.get(&joint).or_else(|| {
                let mut parent = avatar
                    .global_skeleton
                    .joints
                    .get(&joint)
                    .and_then(|j| j.parent);

                while let Some(parent_joint) = parent {
                    if let Some(node) = joint_to_node.get(&parent_joint) {
                        println!(
                            "Attachment {:?}: falling back to parent {:?}",
                            joint, parent_joint
                        );
                        return Some(node);
                    }

                    parent = avatar
                        .global_skeleton
                        .joints
                        .get(&parent_joint)
                        .and_then(|j| j.parent);
                }

                None
            }) else {
                println!(
                    "No skeleton ancestor found for attachment target {:?}",
                    joint
                );
                continue;
            };

            // Set the attachment's transform rotation and scale relative to the joint.
            let node = &mut self.root.nodes[attachment_node.node.value()];

            node.translation = Some((attachment.position * attachment.scale).into());

            let rotation = attachment.rotation;
            node.rotation = Some(UnitQuaternion([
                rotation.x, rotation.y, rotation.z, rotation.w,
            ]));

            node.scale = Some(attachment.scale.into());

            // Make the joint the attachment's parent.
            self.root.nodes[joint_node.value()]
                .children
                .get_or_insert_with(Vec::new)
                .push(attachment_node.node);
        }

        let ibm_accessor_index = self.add_inverse_bind_matrices(&ibm_matrices);

        let root_joints: Vec<Index<Node>> = bones
            .iter()
            .filter_map(|joint_name| {
                if effective_parents[joint_name].is_none() {
                    joint_to_node.get(joint_name).copied()
                } else {
                    None
                }
            })
            .collect();

        let skin_index = self.root.push(Skin {
            joints: skeleton_nodes,
            inverse_bind_matrices: Some(ibm_accessor_index),
            skeleton: root_joints.first().cloned(),
            extensions: Default::default(),
            extras: Default::default(),
            name: Some("AvatarSkin".to_string()),
        });

        // Attach skin to skinned mesh nodes.
        for node_index in &self.skinned_nodes {
            self.root.nodes[node_index.value()].skin = Some(skin_index);
        }

        let skeleton_root_index = self.root.push(Node {
            name: Some("SkeletonRoot".to_string()),
            children: Some(root_joints),
            ..Default::default()
        });

        self.add_bind_pose_animation(avatar, bones, &joint_to_node, &effective_parents);

        //remove nodes that have attachment data. They're already attached.
        let attachment_nodes: HashSet<Index<Node>> = self
            .attachments
            .iter()
            .map(|attachment| attachment.node)
            .collect();

        let non_skinned_mesh_nodes: Vec<Index<Node>> = self
            .mesh_nodes
            .iter()
            .copied()
            .filter(|idx| !self.skinned_nodes.contains(idx))
            .filter(|idx| !attachment_nodes.contains(idx))
            .collect();

        let scene_root_index = self.root.push(Node {
            name: Some("SceneRoot".to_string()),
            children: Some(
                self.skinned_nodes
                    .iter()
                    .copied()
                    .chain(non_skinned_mesh_nodes)
                    .chain(std::iter::once(skeleton_root_index))
                    .collect(),
            ),
            ..Default::default()
        });

        let rotation = Quat::from_rotation_y(-FRAC_PI_2)
            * Quat::from_rotation_z(FRAC_PI_2)
            * Quat::from_rotation_x(-FRAC_PI_2);

        self.root.nodes[scene_root_index.value()].rotation = Some(UnitQuaternion([
            rotation.x, rotation.y, rotation.z, rotation.w,
        ]));

        self.root.push(Scene {
            name: Some("AvatarScene".to_string()),
            nodes: vec![scene_root_index],
            extensions: Default::default(),
            extras: Default::default(),
        });
    }
}

pub fn build_mesh_gltf(
    object: RenderObject,
    path: PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut builder = GltfBuilder::new("Combined Avatar");
    let name = object.name.clone();
    let mesh_index = builder.add_mesh(object, None);
    builder.add_node_with_mesh(mesh_index, &name);
    builder.finalize_scene("Scene");
    builder.finalize(&path)?;
    Ok(())
}

pub fn build_mesh_y_up(
    object: RenderObject,
    path: PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut builder = GltfBuilder::new("Combined Avatar");
    let name = object.name.clone();
    let mesh_index = builder.add_mesh(object, None);
    builder.add_node_with_mesh(mesh_index, &name);
    builder.rotated_finalize_scene("Scene");
    builder.finalize(&path)?;
    Ok(())
}

pub fn build_mesh_scene_gltf(
    objects: Vec<RenderObject>,
    path: &Path,
) -> Result<(), MetaverseMeshError> {
    let mut builder = GltfBuilder::new("Combined Avatar");

    for object in objects {
        let name = object.name.clone();
        let mesh_index = builder.add_mesh(object, None);
        builder.add_node_with_mesh(mesh_index, &name);
    }

    builder.finalize_scene("Scene");
    builder.finalize(path)?;

    Ok(())
}

pub fn build_skinned_mesh_gltf(
    avatar: AvatarObject,
    path: PathBuf,
) -> Result<(), MetaverseMeshError> {
    let mut builder = GltfBuilder::new("Combined Avatar");

    for object in &avatar.objects {
        let json_str = fs::read_to_string(object).map_err(|e| {
            eprintln!("Failed to read object: {:?}: {:?}", object, e);
            e
        })?;

        let parts: RenderObject = serde_json::from_str(&json_str)?;

        builder.add_mesh(parts, Some(&avatar.used_joints));
    }

    builder.finalize_avatar(&avatar);
    builder.finalize(&path)?;

    Ok(())
}

fn find_effective_parents(avatar: &AvatarObject) -> HashMap<SkinJoint, Option<SkinJoint>> {
    let bones = &avatar.used_joints;
    let mut effective_parents = HashMap::new();

    for joint_name in bones {
        let Some(joint) = avatar.global_skeleton.joints.get(joint_name) else {
            continue;
        };

        let mut parent = joint.parent;

        while let Some(parent_name) = parent {
            if bones.contains(&parent_name) {
                effective_parents.insert(*joint_name, Some(parent_name));
                break;
            }

            parent = avatar
                .global_skeleton
                .joints
                .get(&parent_name)
                .and_then(|j| j.parent);
        }

        effective_parents.entry(*joint_name).or_insert(None);
    }

    effective_parents
}
#[cfg(feature = "dxt1")]
fn dxt1_dds_path(input_path: &Path) -> PathBuf {
    let stem = input_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("texture");

    input_path.with_file_name(format!("{stem}.dxt1.dds"))
}

#[cfg(feature = "dxt1")]
pub fn png_to_dxt1_dds(input_path: &Path, output_path: &Path) -> std::io::Result<()> {
    GltfBuilder::png_to_dxt1_dds(input_path, output_path)
}

#[cfg(feature = "ktx2")]
pub fn png_to_ktx2(input_path: &Path, output_path: &Path) -> KtxResult<()> {
    // Decode PNG.
    let image = image::open(input_path)
        .map_err(|_| ErrorCode::FileDataError)?
        .to_rgba8();

    let (width, height) = image.dimensions();
    let pixels = image.into_raw();

    // Create texture description.
    let create_info = TextureCreateInfo {
        vk_format: VkFormat::R8G8B8A8_UNORM,
        base_width: width,
        base_height: height,
        base_depth: 1,
        num_dimensions: 2,
        num_levels: 1,
        num_layers: 1,
        num_faces: 1,
        is_array: false,
        generate_mipmaps: false,
        ..Default::default()
    };

    // Allocate KTX2 texture storage.
    let mut texture = Texture2::create(&create_info, TextureCreateStorage::AllocStorage)?;

    // Copy RGBA pixel data into allocated storage.
    texture
        .get_data_mut()
        .ok_or(ErrorCode::InvalidOperation)?
        .copy_from_slice(&pixels);

    // Configure UASTC encoding.
    let params = BasisParams {
        struct_size: KTX_BASIS_PARAMS_SIZE,
        codec: KTX_BASIS_CODEC_UASTC_LDR_4X4,
        thread_count: 1,
        ..Default::default()
    };

    // Encode using Basis Universal.
    texture.compress_basis_ex(&params)?;

    // Write KTX2.
    texture.write_to_named_file(output_path)?;

    Ok(())
}

/// Converts a byte vector to a vector aligned to a mutiple of 4
fn to_padded_byte_vector(data: &[Vec3]) -> Vec<u8> {
    let flat: Vec<[f32; 3]> = data.iter().map(|v| [v.x, v.y, v.z]).collect();
    let byte_slice: &[u8] = bytemuck::cast_slice(&flat);
    let mut new_vec: Vec<u8> = byte_slice.to_owned();

    while !new_vec.len().is_multiple_of(4) {
        new_vec.push(0); // pad to multiple of four bytes
    }

    new_vec
}

/// determines the highest and lowest points on the mesh to store as min and max
///fn bounding_coords(points: &[Vec3]) -> ([f32; 3], [f32; 3]) {
fn bounding_coords(points: &[Vec3]) -> ([f32; 3], [f32; 3]) {
    let mut min = [f32::MAX, f32::MAX, f32::MAX];
    let mut max = [f32::MIN, f32::MIN, f32::MIN];

    for p in points {
        for i in 0..3 {
            min[i] = f32::min(min[i], p[i]);
            max[i] = f32::max(max[i], p[i]);
        }
    }
    (min, max)
}
