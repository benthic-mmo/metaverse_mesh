use benthic_asset_pipeline::generated::DEFAULT_SKELETON;
use benthic_protocol::{
    default_animations::{AnimationClip, JointAnimation},
    skeleton::{JointName, Skeleton},
};
use glam::{Mat4, Quat, Vec3};
use gltf::{accessor::DataType, animation::Interpolation, binary::Glb};
use gltf_json::{
    self,
    accessor::GenericComponentType,
    animation::{Channel, Target},
    buffer::View,
    scene::UnitQuaternion,
    validation::{Checked, USize64},
    Accessor, Index, Node, Value,
};
use std::{
    borrow::Cow,
    collections::{BTreeSet, HashMap, HashSet},
    fs::File,
    path::PathBuf,
};

pub struct GltfBuilder {
    pub root: gltf_json::Root,
    buffer_index: Index<gltf_json::Buffer>,
    combined_buffer: Vec<u8>,
    joint_to_node: HashMap<JointName, Index<Node>>,
}

impl GltfBuilder {
    pub fn new(buffer_name: &str) -> Self {
        let mut root = gltf_json::Root::default();
        let buffer_index = root.push(gltf_json::Buffer {
            byte_length: USize64::from(0_usize),
            name: Some(buffer_name.to_string()),
            uri: None,
            extensions: Default::default(),
            extras: Default::default(),
        });
        Self {
            root,
            buffer_index,
            combined_buffer: Vec::new(),
            joint_to_node: HashMap::new(),
        }
    }
    pub fn add_skin_from_skeleton(
        &mut self,
        skeleton: &Skeleton,
        joint_filter: &BTreeSet<JointName>,
    ) {
        let mut joint_nodes = Vec::new();
        let mut ibm_matrices = Vec::new();

        for joint_name in joint_filter {
            let joint = &skeleton.joints[joint_name];
            let node_index = self.joint_to_node[joint_name];

            joint_nodes.push(node_index);

            let ibm = joint.transforms[0].transform.inverse();
            ibm_matrices.push(ibm.to_cols_array());
        }

        let ibm_accessor = self.push_mat4_accessor_flat(&ibm_matrices, "ibm");

        let root_joint = joint_filter
            .iter()
            .find(|jn| skeleton.joints[*jn].parent.is_none())
            .map(|jn| self.joint_to_node[jn]);

        let _skin_index = self.root.push(gltf_json::Skin {
            joints: joint_nodes.clone(),
            inverse_bind_matrices: Some(ibm_accessor),
            skeleton: root_joint, // the skeleton root node
            name: Some("SkeletonRoot".to_string()),
            extensions: Default::default(),
            extras: Default::default(),
        });
    }

    fn push_mat4_accessor_flat(&mut self, values: &[[f32; 16]], name: &str) -> Index<Accessor> {
        self.align_4();
        let offset = self.combined_buffer.len();

        for mat in values {
            for f in mat {
                self.combined_buffer.extend_from_slice(&f.to_le_bytes());
            }
        }

        let view = self.root.push(View {
            buffer: self.buffer_index,
            byte_length: (values.len() * 16 * 4).into(),
            byte_offset: Some(USize64(offset as u64)),
            byte_stride: None,
            target: None,
            name: Some(name.to_string()),
            extensions: Default::default(),
            extras: Default::default(),
        });

        self.root.push(Accessor {
            buffer_view: Some(view),
            byte_offset: Some(USize64(0)),
            count: USize64(values.len() as u64),
            component_type: Checked::Valid(GenericComponentType(DataType::F32)),
            type_: Checked::Valid(gltf_json::accessor::Type::Mat4),
            normalized: false,
            min: None,
            max: None,
            sparse: None,
            name: Some(name.to_string()),
            extensions: Default::default(),
            extras: Default::default(),
        })
    }

    fn align_4(&mut self) {
        while self.combined_buffer.len() % 4 != 0 {
            self.combined_buffer.push(0);
        }
    }

    fn push_scalar_accessor(&mut self, values: &[f32], name: &str) -> Index<Accessor> {
        self.align_4();
        let offset = self.combined_buffer.len();
        for &v in values {
            self.combined_buffer.extend_from_slice(&v.to_le_bytes());
        }

        let min = values.iter().copied().fold(f32::INFINITY, f32::min);
        let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        let view = self.root.push(View {
            buffer: self.buffer_index,
            byte_length: (values.len() * 4).into(),
            byte_offset: Some(USize64(offset as u64)),
            byte_stride: None,
            target: None,
            name: Some(name.to_string()),
            extensions: Default::default(),
            extras: Default::default(),
        });

        self.root.push(Accessor {
            buffer_view: Some(view),
            byte_offset: Some(USize64(0)),
            count: USize64(values.len() as u64),
            component_type: Checked::Valid(GenericComponentType(DataType::F32)),
            type_: Checked::Valid(gltf_json::accessor::Type::Scalar),
            normalized: false,
            min: Some(Value::from(vec![min])),
            max: Some(Value::from(vec![max])),
            sparse: None,
            name: Some(name.to_string()),
            extensions: Default::default(),
            extras: Default::default(),
        })
    }

    fn push_vec3_accessor(&mut self, values: &[[f32; 3]], name: &str) -> Index<Accessor> {
        self.align_4();
        let offset = self.combined_buffer.len();
        for v in values {
            for f in v {
                self.combined_buffer.extend_from_slice(&f.to_le_bytes());
            }
        }
        let min = [
            values.iter().map(|v| v[0]).fold(f32::INFINITY, f32::min),
            values.iter().map(|v| v[1]).fold(f32::INFINITY, f32::min),
            values.iter().map(|v| v[2]).fold(f32::INFINITY, f32::min),
        ];
        let max = [
            values
                .iter()
                .map(|v| v[0])
                .fold(f32::NEG_INFINITY, f32::max),
            values
                .iter()
                .map(|v| v[1])
                .fold(f32::NEG_INFINITY, f32::max),
            values
                .iter()
                .map(|v| v[2])
                .fold(f32::NEG_INFINITY, f32::max),
        ];

        let view = self.root.push(View {
            buffer: self.buffer_index,
            byte_length: (values.len() * 3 * 4).into(),
            byte_offset: Some(USize64(offset as u64)),
            byte_stride: None,
            target: None,
            name: Some(name.to_string()),
            extensions: Default::default(),
            extras: Default::default(),
        });

        self.root.push(Accessor {
            buffer_view: Some(view),
            byte_offset: Some(USize64(0)),
            count: USize64(values.len() as u64),
            component_type: Checked::Valid(GenericComponentType(DataType::F32)),
            type_: Checked::Valid(gltf_json::accessor::Type::Vec3),
            normalized: false,
            min: Some(Value::from(min.to_vec())),
            max: Some(Value::from(max.to_vec())),
            sparse: None,
            name: Some(name.to_string()),
            extensions: Default::default(),
            extras: Default::default(),
        })
    }

    fn push_vec4_accessor(&mut self, values: &[[f32; 4]], name: &str) -> Index<Accessor> {
        self.align_4();
        let offset = self.combined_buffer.len();
        for v in values {
            for f in v {
                self.combined_buffer.extend_from_slice(&f.to_le_bytes());
            }
        }
        let min = [
            values.iter().map(|v| v[0]).fold(f32::INFINITY, f32::min),
            values.iter().map(|v| v[1]).fold(f32::INFINITY, f32::min),
            values.iter().map(|v| v[2]).fold(f32::INFINITY, f32::min),
            values.iter().map(|v| v[3]).fold(f32::INFINITY, f32::min),
        ];
        let max = [
            values
                .iter()
                .map(|v| v[0])
                .fold(f32::NEG_INFINITY, f32::max),
            values
                .iter()
                .map(|v| v[1])
                .fold(f32::NEG_INFINITY, f32::max),
            values
                .iter()
                .map(|v| v[2])
                .fold(f32::NEG_INFINITY, f32::max),
            values
                .iter()
                .map(|v| v[3])
                .fold(f32::NEG_INFINITY, f32::max),
        ];

        let view = self.root.push(View {
            buffer: self.buffer_index,
            byte_length: (values.len() * 4 * 4).into(),
            byte_offset: Some(USize64(offset as u64)),
            byte_stride: None,
            target: None,
            name: Some(name.to_string()),
            extensions: Default::default(),
            extras: Default::default(),
        });

        self.root.push(Accessor {
            buffer_view: Some(view),
            byte_offset: Some(USize64(0)),
            count: USize64(values.len() as u64),
            component_type: Checked::Valid(GenericComponentType(DataType::F32)),
            type_: Checked::Valid(gltf_json::accessor::Type::Vec4),
            normalized: false,
            min: Some(Value::from(min.to_vec())),
            max: Some(Value::from(max.to_vec())),
            sparse: None,
            name: Some(name.to_string()),
            extensions: Default::default(),
            extras: Default::default(),
        })
    }

    pub fn add_filtered_animation(
        &mut self,
        skeleton: &Skeleton,
        animations: &[JointAnimation],
        joint_filter: &BTreeSet<JointName>,
    ) {
        // First push all nodes
        for (joint_name, joint) in skeleton.joints.iter() {
            if !joint_filter.contains(joint_name) {
                continue;
            }
            let local = joint.local_transforms[0].transform;

            let (scale, rotation, translation) = local.to_scale_rotation_translation();
            let node_index = self.root.push(Node {
                name: Some(joint_name.to_string()),
                translation: Some([translation.x, translation.y, translation.z]),
                rotation: Some(UnitQuaternion([
                    rotation.x, rotation.y, rotation.z, rotation.w,
                ])),
                scale: Some([scale.x, scale.y, scale.z]),
                ..Default::default()
            });

            self.joint_to_node.insert(*joint_name, node_index);
        }

        for (joint_name, joint) in skeleton.joints.iter() {
            if !joint_filter.contains(joint_name) {
                continue;
            }
            let parent_node = self.joint_to_node[joint_name];
            let children_indices: Vec<_> = joint
                .children
                .iter()
                .filter(|c| joint_filter.contains(c))
                .map(|c| self.joint_to_node[c])
                .collect();
            if !children_indices.is_empty() {
                self.root.nodes[parent_node.value() as usize].children = Some(children_indices);
            }
        }

        let mut samplers = Vec::new();
        let mut channels = Vec::new();

        for joint_anim in animations
            .iter()
            .filter(|a| joint_filter.contains(&a.joint))
        {
            let node_index = self.joint_to_node[&joint_anim.joint];

            if !joint_anim.translations.is_empty() {
                if !joint_anim.translations.is_empty() {
                    let times: Vec<f32> = joint_anim.translations.iter().map(|k| k.time).collect();

                    let values: Vec<[f32; 3]> = joint_anim
                        .translations
                        .iter()
                        .map(|k| {
                            let v = k.value;
                            [v.x, v.y, v.z]
                        })
                        .collect();

                    let input_accessor = self.push_scalar_accessor(
                        &times,
                        &format!("{:?}_translation_times", joint_anim.joint),
                    );

                    let output_accessor = self.push_vec3_accessor(
                        &values,
                        &format!("{:?}_translation_values", joint_anim.joint),
                    );

                    let sampler_index = samplers.len() as u32;

                    samplers.push(gltf_json::animation::Sampler {
                        input: input_accessor,
                        output: output_accessor,
                        interpolation: Checked::Valid(Interpolation::Linear),
                        extras: Default::default(),
                        extensions: None,
                    });

                    channels.push(Channel {
                        sampler: Index::new(sampler_index),
                        target: Target {
                            node: node_index,
                            path: Checked::Valid(gltf::animation::Property::Translation),
                            extras: Default::default(),
                            extensions: None,
                        },
                        extras: Default::default(),
                        extensions: None,
                    });
                }
            }

            if !joint_anim.rotations.is_empty() {
                let times: Vec<f32> = joint_anim.rotations.iter().map(|k| k.time).collect();

                let values: Vec<[f32; 4]> = joint_anim
                    .rotations
                    .iter()
                    .map(|k| {
                        let q = k.value;
                        [q.x, q.y, q.z, q.w]
                    })
                    .collect();

                let input_accessor = self.push_scalar_accessor(
                    &times,
                    &format!("{:?}_rotation_times", joint_anim.joint),
                );

                let output_accessor = self.push_vec4_accessor(
                    &values,
                    &format!("{:?}_rotation_values", joint_anim.joint),
                );

                let sampler_index = samplers.len() as u32;

                samplers.push(gltf_json::animation::Sampler {
                    input: input_accessor,
                    output: output_accessor,
                    interpolation: Checked::Valid(Interpolation::Linear),
                    extras: Default::default(),
                    extensions: None,
                });

                channels.push(Channel {
                    sampler: Index::new(sampler_index),
                    target: Target {
                        node: node_index,
                        path: Checked::Valid(gltf::animation::Property::Rotation),
                        extras: Default::default(),
                        extensions: None,
                    },
                    extras: Default::default(),
                    extensions: None,
                });
            }

            if !joint_anim.scales.is_empty() {
                let times: Vec<f32> = joint_anim.scales.iter().map(|k| k.time).collect();

                let values: Vec<[f32; 3]> = joint_anim
                    .scales
                    .iter()
                    .map(|k| {
                        let v = k.value;
                        [v.x, v.y, v.z]
                    })
                    .collect();

                let input_accessor = self
                    .push_scalar_accessor(&times, &format!("{:?}_scale_times", joint_anim.joint));

                let output_accessor = self
                    .push_vec3_accessor(&values, &format!("{:?}_scale_values", joint_anim.joint));

                let sampler_index = samplers.len() as u32;

                samplers.push(gltf_json::animation::Sampler {
                    input: input_accessor,
                    output: output_accessor,
                    interpolation: Checked::Valid(Interpolation::Linear),
                    extras: Default::default(),
                    extensions: None,
                });

                channels.push(Channel {
                    sampler: Index::new(sampler_index),
                    target: Target {
                        node: node_index,
                        path: Checked::Valid(gltf::animation::Property::Scale),
                        extras: Default::default(),
                        extensions: None,
                    },
                    extras: Default::default(),
                    extensions: None,
                });
            }
        }

        if !samplers.is_empty() {
            self.root.animations.push(gltf_json::Animation {
                samplers,
                channels,
                name: Some("filtered_animation".to_string()),
                extras: Default::default(),
                extensions: None,
            });
        }
    }

    pub fn finalize_scene(&mut self, name: &str) {
        // find all child nodes
        let mut children_set = HashSet::new();
        for node in self.root.nodes.iter() {
            if let Some(children) = &node.children {
                children_set.extend(children.iter().map(|c| c.value()));
            }
        }

        let top_level_joints: Vec<_> = self
            .joint_to_node
            .values()
            .filter(|n| !children_set.contains(&n.value()))
            .copied()
            .collect();

        let skeleton_root_index = self.root.push(Node {
            name: Some("SkeletonRoot".to_string()),
            translation: Some([0.0, 0.0, 0.0]),
            rotation: Some(UnitQuaternion([0.0, 0.0, 0.0, 1.0])),
            scale: Some([1.0, 1.0, 1.0]),
            children: Some(top_level_joints),
            mesh: None,
            skin: None,
            extensions: Default::default(),
            extras: Default::default(),
            ..Default::default()
        });

        let scene_root_index = self.root.push(Node {
            name: Some("SceneRoot".to_string()),
            translation: Some([0.0, 0.0, 0.0]),
            rotation: Some(UnitQuaternion([0.0, 0.0, 0.0, 1.0])),
            scale: Some([1.0, 1.0, 1.0]),
            children: Some(vec![skeleton_root_index]),
            mesh: None,
            skin: None,
            extensions: Default::default(),
            extras: Default::default(),
            ..Default::default()
        });

        self.root.push(gltf_json::Scene {
            name: Some(name.to_string()),
            nodes: vec![scene_root_index],
            extensions: Default::default(),
            extras: Default::default(),
        });
    }

    pub fn finalize(&mut self, path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
        self.root.buffers[self.buffer_index.value() as usize].byte_length =
            USize64::from(self.combined_buffer.len());
        let json_bytes = gltf_json::serialize::to_string(&self.root)?.into_bytes();
        let glb = Glb {
            header: gltf::binary::Header {
                magic: *b"glTF",
                version: 2,
                length: (json_bytes.len() + self.combined_buffer.len()) as u32,
            },
            json: Cow::Owned(json_bytes),
            bin: Some(Cow::Owned(std::mem::take(&mut self.combined_buffer))),
        };
        glb.to_writer(File::create(path)?)?;
        Ok(())
    }
}

pub fn export_filtered_animation(
    animations: &AnimationClip,
    joint_filter: &BTreeSet<JointName>,
    out_path: PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let skeleton: Skeleton = DEFAULT_SKELETON.clone();
    let mut builder = GltfBuilder::new("animation");
    builder.add_filtered_animation(&skeleton, &animations.joints, joint_filter);
    builder.add_skin_from_skeleton(&skeleton, joint_filter);
    builder.finalize_scene("filtered_scene");
    builder.finalize(&out_path)
}

pub fn export_animation_clip(
    clip: &AnimationClip,
    out_path: PathBuf,
    skeleton: Skeleton,
) -> Result<(), Box<dyn std::error::Error>> {
    let joint_filter: BTreeSet<JointName> = clip.joints.iter().map(|j| j.joint).collect();

    let skeleton: Skeleton = DEFAULT_SKELETON.clone();
    let mut builder = GltfBuilder::new("animation");
    check_skeleton_cycles(&skeleton)?;

    builder.add_filtered_animation(&skeleton, &clip.joints, &joint_filter);

    builder.add_skin_from_skeleton(&skeleton, &joint_filter);

    builder.finalize_scene("animation_scene");

    builder.finalize(&out_path)?;

    Ok(())
}

fn check_skeleton_cycles(skeleton: &Skeleton) -> Result<(), String> {
    fn visit(
        joint: JointName,
        skeleton: &Skeleton,
        visiting: &mut HashSet<JointName>,
        visited: &mut HashSet<JointName>,
    ) -> Result<(), String> {
        if visiting.contains(&joint) {
            return Err(format!("Skeleton cycle detected at {:?}", joint));
        }

        if visited.contains(&joint) {
            return Ok(());
        }

        visiting.insert(joint);

        if let Some(node) = skeleton.joints.get(&joint) {
            for child in &node.children {
                visit(*child, skeleton, visiting, visited)?;
            }
        }

        visiting.remove(&joint);
        visited.insert(joint);

        Ok(())
    }

    let mut visiting = HashSet::new();
    let mut visited = HashSet::new();

    for joint in skeleton.joints.keys() {
        visit(*joint, skeleton, &mut visiting, &mut visited)?;
    }

    Ok(())
}
