# Metaverse GLTF
[![last-commit][last-commit-badge]][last-commit] [![open-pr][open-pr-badge]][open-pr] [![open-issues][open-issues-badge]][open-issues]

This is a simple crate that converts JSON serialized SceneObject data to GLTF files that can be rendered by metaverse clients and servers. This crate is a work in progress, but hopefully will be useful in creating optional GLTF-serving endpoints for metaverse servers, and allowing other metaverse client projects to have direct access to easier to use 3d models.

Ideally, the interface for generating viewable, debuggable 3d models will receive nothing but JSON input and output a filename, allowing viewer backends to be fully format agnostic, allowing their rendering programs to handle the rendering of the files independently.

[docs.rs-badge]: https://img.shields.io/badge/docs-Docs.rs-red?&style=flat-square

[last-commit-badge]:https://img.shields.io/github/last-commit/benthic-mmo/metaverse_gltf?logo=github&style=flat-square
[last-commit]: https://github.com/benthic-mmo/metaverse_gltf/commits/main/

[open-pr-badge]:https://img.shields.io/github/issues-pr/benthic-mmo/metaverse_gltf?logo=github&style=flat-square
[open-pr]: https://github.com/benthic-mmo/metaverse_gltf/pulls

[open-issues-badge]:https://img.shields.io/github/issues-raw/benthic-mmo/metaverse_gltf?logo=github&style=flat-square
[open-issues]: https://github.com/benthic-mmo/metaverse_gltf/issues
[docs.rs-badge]: https://img.shields.io/badge/docs-Docs.rs-red?&style=flat-square

[crates.io-session-badge]: https://img.shields.io/crates/v/metaverse_session?logo=rust&logoColor=white&style=flat-square
[crates.io-session]: https://crates.io/crates/metaverse_messages
[docs.rs-session]: https://docs.rs/metaverse_session/latest/metaverse_session/

[crates.io-messages-badge]: https://img.shields.io/crates/v/metaverse_messages?logo=rust&logoColor=white&style=flat-square
[crates.io-messages]: https://crates.io/crates/metaverse_messages
[docs.rs-messages]: https://docs.rs/metaverse_messages/latest/metaverse_session/

[last-commit-badge]:https://img.shields.io/github/last-commit/benthic-mmo/metaverse_client?logo=github&style=flat-square
[last-commit]: https://github.com/benthic-mmo/metaverse_client/commits/main/

[open-pr-badge]:https://img.shields.io/github/issues-pr/benthic-mmo/metaverse_client?logo=github&style=flat-square
[open-pr]: https://github.com/benthic-mmo/metaverse_client/pulls
