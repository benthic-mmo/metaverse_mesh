use benthic_protocol::render_data::{AvatarObject, JointWeight, RenderObject};
use benthic_protocol::skeleton::{JointName, SkinJoint};
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
use rgb::bytemuck;
use std::f32::consts::FRAC_PI_2;
use std::{
    borrow::Cow,
    collections::{BTreeSet, HashMap},
    fs::{self, File},
    path::PathBuf,
    vec,
};

use crate::errors::MetaverseMeshError;

struct GltfBuilder {
    root: gltf_json::Root,
    buffer_index: gltf_json::Index<gltf_json::Buffer>,
    combined_buffer: Vec<u8>,
    nodes: Vec<Index<Node>>,
    mesh_nodes: Vec<Index<Node>>,
    skinned_nodes: Vec<Index<Node>>,
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
        bones: &BTreeSet<JointName>,
        joint_to_node: &HashMap<JointName, Index<Node>>,
        effective_parents: &HashMap<JointName, Option<JointName>>,
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
        bones: &BTreeSet<JointName>,
    ) -> (Option<Index<Accessor>>, Option<Index<Accessor>>) {
        if skin_weights.is_empty() {
            return (None, None);
        }

        let bone_index: HashMap<JointName, u8> = bones
            .iter()
            .enumerate()
            .map(|(i, j)| (*j, i as u8))
            .collect();

        let mut joint_indices_bytes = Vec::new();
        let mut joint_weights_bytes = Vec::new();

        for vw in &skin_weights {
            let mut joints = [0u8; 4];
            let mut weights = [0.0f32; 4];

            // only process Joints, and not collision joints.
            // TODO: Process collision joints.
            for i in 0..4 {
                if let (Some(skin_joint), Some(&weight)) = (vw.joint_name.get(i), vw.weights.get(i))
                    && weight > 0.0
                    && let SkinJoint::Joint(joint_name) = skin_joint
                    && let Some(&idx) = bone_index.get(joint_name)
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
        used_joints: Option<&BTreeSet<JointName>>,
    ) -> gltf_json::Index<gltf_json::Mesh> {
        let pos_accessor = self.add_vertex_positions(&object.vertices);
        let index_accessor = self.add_indices(&object.indices);

        let mut attributes: std::collections::BTreeMap<_, _> =
            [(Valid(Semantic::Positions), pos_accessor)]
                .into_iter()
                .collect();

        if let Some(normals) = object.normals {
            assert_eq!(
                normals.len(),
                object.vertices.len(),
                "Normal count ({}) does not match position count ({})",
                normals.len(),
                object.vertices.len()
            );

            let normal_accessor = self.add_normals(&normals);
            attributes.insert(Valid(Semantic::Normals), normal_accessor);
        }

        let material = if let Some(uvs) = object.uv {
            let uv_accessor = self.add_uvs(&uvs);
            attributes.insert(Valid(Semantic::TexCoords(0)), uv_accessor);

            if let Some(texture_path) = object.texture {
                let (_, _, material_index) = self.add_texture(&texture_path);
                Some(material_index)
            } else {
                None
            }
        } else {
            None
        };

        if let Some(skin) = &object.skin
            && let Some(used_joints) = used_joints
        {
            let (joint_indices_accessor, joint_weights_accessor) =
                self.add_joint_data(skin.weights.clone(), used_joints);
            if let Some(joints) = joint_indices_accessor {
                attributes.insert(Valid(Semantic::Joints(0)), joints);
            }
            if let Some(weights) = joint_weights_accessor {
                attributes.insert(Valid(Semantic::Weights(0)), weights);
            }
        };

        let primitive = Primitive {
            attributes,
            indices: Some(index_accessor),
            material,
            mode: Valid(Mode::Triangles),
            targets: None,
            extensions: Default::default(),
            extras: Default::default(),
        };

        let mesh_index = self.root.push(Mesh {
            primitives: vec![primitive],
            weights: None,
            extensions: Default::default(),
            extras: Default::default(),
            name: Some(object.name.clone()),
        });

        let node_index = self.add_node_with_mesh(mesh_index, &object.name);

        self.mesh_nodes.push(node_index);

        if object.skin.is_some() {
            self.skinned_nodes.push(node_index);
        }

        mesh_index
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

    pub fn add_texture(
        &mut self,
        image_path: &PathBuf,
    ) -> (
        Index<gltf_json::Image>,
        Index<gltf_json::Texture>,
        Index<gltf_json::Material>,
    ) {
        let image_data = fs::read(image_path).unwrap_or_else(|e| {
            panic!("Failed to read image file {}: {}", image_path.display(), e);
        });

        self.align_4();

        let buffer_byte_offset = self.combined_buffer.len() as u64;
        self.combined_buffer.extend_from_slice(&image_data);

        let buffer_view_index = self.root.push(View {
            buffer: Index::new(0),
            byte_length: image_data.len().into(),
            byte_offset: Some(USize64(buffer_byte_offset)),
            byte_stride: None,
            target: None,
            name: Some("image_buffer_view".to_string()),
            extensions: Default::default(),
            extras: Default::default(),
        });

        let mime_type = match image_path.extension().and_then(|s| s.to_str()) {
            Some("png") => MimeType("image/png".to_string()),
            Some("jpg" | "jpeg") => MimeType("image/jpeg".to_string()),
            extension => panic!(
                "Unsupported texture format {:?}: {}",
                extension,
                image_path.display()
            ),
        };

        let image_index = self.root.push(gltf_json::Image {
            uri: None,
            mime_type: Some(mime_type),
            buffer_view: Some(buffer_view_index),
            name: Some("diffuse".to_string()),
            extensions: Default::default(),
            extras: Default::default(),
        });

        let texture_index = self.root.push(gltf_json::Texture {
            sampler: None,
            source: image_index,
            name: Some("diffuse_texture".to_string()),
            extensions: Default::default(),
            extras: Default::default(),
        });

        let material_index = self.root.push(Material {
            pbr_metallic_roughness: PbrMetallicRoughness {
                base_color_texture: Some(texture::Info {
                    index: texture_index,
                    tex_coord: 0,
                    extensions: Default::default(),
                    extras: Default::default(),
                }),
                metallic_factor: StrengthFactor(0.0),
                roughness_factor: StrengthFactor(1.0),
                ..Default::default()
            },
            alpha_mode: Checked::Valid(AlphaMode::Mask),
            name: Some("material_with_texture".to_string()),
            ..Default::default()
        });

        (image_index, texture_index, material_index)
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

    fn finalize(mut self, path: &PathBuf) -> Result<PathBuf, Box<dyn std::error::Error>> {
        self.root.buffers[self.buffer_index.value()].byte_length =
            gltf_json::validation::USize64::from(self.combined_buffer.len());

        let json_string = gltf_json::serialize::to_string(&self.root)?;
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
                path,
                path.exists(),
                path.parent(),
                path.parent().map(|p| p.exists()).unwrap_or(false),
                e,
            )
        })?;

        glb.to_writer(file)?;
        Ok(path.clone())
    }

    pub fn finalize_avatar(&mut self, avatar: &AvatarObject) {
        let bones = &avatar.used_joints;

        if bones.is_empty() {
            self.finalize_scene("AvatarScene");
            return;
        }

        let mut joint_to_node: HashMap<JointName, Index<Node>> = HashMap::new();

        let mut skeleton_nodes = Vec::new();
        let mut ibm_matrices = Vec::new();
        let effective_parents = find_effective_parents(avatar);

        // Create joint nodes.
        for joint_name in bones {
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

        // Put meshes and skeleton underneath one root.
        let non_skinned_mesh_nodes: Vec<Index<Node>> = self
            .mesh_nodes
            .iter()
            .copied()
            .filter(|idx| !self.skinned_nodes.contains(idx))
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
    path: PathBuf,
) -> Result<(), MetaverseMeshError> {
    let mut builder = GltfBuilder::new("Combined Avatar");

    for object in objects {
        let name = object.name.clone();
        let mesh_index = builder.add_mesh(object, None);
        builder.add_node_with_mesh(mesh_index, &name);
    }

    builder.finalize_scene("Scene");
    builder.finalize(&path)?;

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

        let parts: Vec<RenderObject> = serde_json::from_str(&json_str)?;

        for part in parts {
            builder.add_mesh(part, Some(&avatar.used_joints));
        }
    }

    builder.finalize_avatar(&avatar);
    builder.finalize(&path)?;

    Ok(())
}

fn find_effective_parents(avatar: &AvatarObject) -> HashMap<JointName, Option<JointName>> {
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
