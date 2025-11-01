use glam::{usize, Quat, Vec3};
use gltf_json::{
    accessor::{ComponentType, GenericComponentType},
    buffer::{Stride, Target, View},
    material::{PbrMetallicRoughness, StrengthFactor},
    mesh::{Mode, Primitive, Semantic},
    scene::UnitQuaternion,
    texture,
    validation::{
        Checked::{self, Valid},
        USize64,
    },
    Accessor, Index, Material, Mesh, Node, Scene, Value,
};
use metaverse_messages::utils::render_data::RenderObject;
use metaverse_messages::utils::skeleton::JointName;
use metaverse_messages::{http::mesh::JointWeight, utils::render_data::AvatarObject};
use rgb::bytemuck;
use std::{
    borrow::Cow,
    collections::BTreeSet,
    fs::{self, File},
    path::PathBuf,
    vec,
};

struct GltfBuilder {
    root: gltf_json::Root,
    buffer_index: gltf_json::Index<gltf_json::Buffer>,
    combined_buffer: Vec<u8>,
    nodes: Vec<Index<Node>>,
}

impl GltfBuilder {
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
        }
    }
    fn align_4(&mut self) {
        while self.combined_buffer.len() % 4 != 0 {
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

    fn add_indices(&mut self, indices: &[u16]) -> gltf_json::Index<gltf_json::Accessor> {
        let mut bytes = Vec::with_capacity(indices.len() * 2);
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
            component_type: Valid(GenericComponentType(ComponentType::U16)),
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

    /// Adds joint indices and weights for a skinned mesh, returns (indices_accessor, weights_accessor)
    fn add_joint_data(
        &mut self,
        skin_weights: Vec<JointWeight>,
        bones: &BTreeSet<metaverse_messages::utils::skeleton::JointName>,
    ) -> (Index<Accessor>, Index<Accessor>) {
        let mut joint_indices_bytes = Vec::new();
        let mut joint_weights_bytes = Vec::new();

        for vw in &skin_weights {
            let joints: Vec<u8> = vw
                .joint_name
                .iter()
                .filter_map(|j| {
                    bones.iter().enumerate().find_map(|(i, joint_name)| {
                        if joint_name == j {
                            Some(i as u8)
                        } else {
                            None
                        }
                    })
                })
                .collect();

            for (&joint, &weight) in joints.iter().zip(&vw.weights) {
                joint_indices_bytes.push(joint);
                joint_weights_bytes.extend_from_slice(&weight.to_le_bytes());
            }
        }

        while self.combined_buffer.len() % 4 != 0 {
            self.combined_buffer.push(0);
        }
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
            name: Some("joint_indices".to_string()),
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
            name: Some("JOINTS".to_string()),
            min: None,
            max: None,
        });

        while self.combined_buffer.len() % 4 != 0 {
            self.combined_buffer.push(0);
        }
        let weights_offset = self.combined_buffer.len();
        self.combined_buffer.extend_from_slice(&joint_weights_bytes);
        let weights_view = self.root.push(View {
            buffer: self.buffer_index,
            byte_length: USize64::from(joint_weights_bytes.len()),
            byte_offset: Some(USize64::from(weights_offset)),
            byte_stride: Some(Stride(4 * std::mem::size_of::<f32>())),
            target: Some(Checked::Valid(Target::ArrayBuffer)),
            extensions: None,
            extras: Default::default(),
            name: Some("joint_weights".to_string()),
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
            name: Some("WEIGHTS".to_string()),
            min: None,
            max: None,
        });

        (indices_accessor, weights_accessor)
    }

    pub fn add_mesh(
        &mut self,
        name: &str,
        positions: &[Vec3],
        indices: &[u16],
    ) -> gltf_json::Index<gltf_json::Mesh> {
        let pos_accessor = self.add_vertex_positions(positions);
        let index_accessor = self.add_indices(indices);

        let attributes = [(Valid(Semantic::Positions), pos_accessor)]
            .into_iter()
            .collect();

        let primitive = Primitive {
            attributes,
            indices: Some(index_accessor),
            material: None,
            mode: Valid(Mode::Triangles),
            targets: None,
            extensions: Default::default(),
            extras: Default::default(),
        };

        self.root.push(Mesh {
            primitives: vec![primitive],
            weights: None,
            extensions: Default::default(),
            extras: Default::default(),
            name: Some(name.to_string()),
        })
    }

    pub fn add_node_with_mesh(
        &mut self,
        mesh_index: gltf_json::Index<gltf_json::Mesh>,
        name: &str,
    ) {
        let node_index = self.root.push(Node {
            mesh: Some(mesh_index),
            name: Some(name.to_string()),
            ..Default::default()
        });
        self.nodes.push(node_index);
    }
    pub fn add_uvs(&mut self, uvs: &[[f32; 2]]) -> gltf_json::Index<gltf_json::Accessor> {
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
        image_path: &str,
    ) -> (
        gltf_json::Index<gltf_json::Image>,
        gltf_json::Index<gltf_json::Texture>,
        gltf_json::Index<gltf_json::Material>,
    ) {
        let image_index = self.root.push(gltf_json::Image {
            uri: Some(image_path.to_string()),
            mime_type: None,
            buffer_view: None,
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
            name: Some("material_with_texture".to_string()),
            ..Default::default()
        });

        (image_index, texture_index, material_index)
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

    fn finalize(mut self, path: &PathBuf) -> Result<PathBuf, Box<dyn std::error::Error>> {
        self.root.buffers[self.buffer_index.value() as usize].byte_length =
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
        glb.to_writer(File::create(&path)?)?;
        Ok(path.clone())
    }
}

pub fn build_mesh_gltf(
    object: RenderObject,
    path: PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut builder = GltfBuilder::new("Combined Avatar");
    let mesh_index = builder.add_mesh(&object.name, &object.vertices, &object.indices);
    builder.add_node_with_mesh(mesh_index, &object.name);

    builder.add_uvs(&object.uv.unwrap());
    builder.finalize_scene(&format!("Scene"));
    builder.finalize(&path)?;
    Ok(())
}

pub fn build_mesh_y_up(
    object: RenderObject,
    path: PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut builder = GltfBuilder::new("Combined Avatar");
    let mesh_index = builder.add_mesh(&object.name, &object.vertices, &object.indices);
    builder.add_node_with_mesh(mesh_index, &object.name);
    builder.rotated_finalize_scene(&format!("Scene"));
    builder.finalize(&path)?;
    Ok(())
}

pub fn build_mesh_scene_gltf(
    objects: Vec<RenderObject>,
    path: PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut builder = GltfBuilder::new("Combined Avatar");
    for object in objects {
        let mesh_index = builder.add_mesh(&object.name, &object.vertices, &object.indices);
        builder.add_node_with_mesh(mesh_index, &object.name);
    }
    builder.finalize_scene(&format!("Scene"));
    builder.finalize(&path)?;
    Ok(())
}

pub fn build_skinned_mesh_gltf(
    avatar: AvatarObject,
    path: PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut bones: BTreeSet<JointName> = BTreeSet::new();
    let mut builder = GltfBuilder::new("Combined Avatar");

    for joint in &avatar.global_skeleton.joints {
        if joint.1.transforms.len() > 1 {
            bones.insert(*joint.0);
        }
    }

    for object in avatar.objects {
        let json_str = fs::read_to_string(&object)?;
        let parts: Vec<RenderObject> = serde_json::from_str(&json_str)?;

        for part in parts {
            let mesh_index = builder.add_mesh(&part.name, &part.vertices, &part.indices);
            builder.add_node_with_mesh(mesh_index, &part.name);
            builder.add_joint_data(part.skin.unwrap().weights, &bones);
        }
    }
    builder.rotated_finalize_scene(&format!("Scene"));
    builder.finalize(&path)?;
    Ok(())
}
/// Converts a byte vector to a vector aligned to a mutiple of 4
fn to_padded_byte_vector(data: &[Vec3]) -> Vec<u8> {
    let flat: Vec<[f32; 3]> = data.iter().map(|v| [v.x, v.y, v.z]).collect();
    let byte_slice: &[u8] = bytemuck::cast_slice(&flat);
    let mut new_vec: Vec<u8> = byte_slice.to_owned();

    while new_vec.len() % 4 != 0 {
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
