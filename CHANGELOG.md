## [Unreleased]
### Added
- Add BVH cost debug kernel to viewer, can be activated by pressing
KEY-4, or by passing -algorithm=costs on the command line.
- Set number of convergence frames rendered in viewer, pause and
resume using Key-Space.
- Experimental short stack BVH traversal with restart trail based on
Laine 2010 HPG paper (only on GPU).
- Phantom ray-hair intersector example demonstrating how to extend
the API with a custom curve primitive type.
- CPU-parallel BVH refitter. Refitting is used in juggler example.
- New "juggler" example demonstrating dynamic scenes and procedural
textures.

### Changed
- Don't use clock as LCG seed anymore; rather compute hashes based on
pixelID and frameID.
- BVH instances now store 4x3 instead of 4x4 transform matrices.
- vec3 is no longer aligned to 16 byte boundaries.
- Switched to Ubuntu 16.04 with Travis-CI. We no longer support
Ubuntu 14.04 and gcc 4.8 as that is too old for pbrtParser.
- pbrtParser is now a submodule instead of an external dependency
supplied by the user.

### Fixed
- Various smaller fixes to make things run smoothly on Apple silicon.
- Fixed linking with homebrew GLEW on macOS.

## [0.1.1] - 2020-06-22
### Fixed
- CUDA LBVH builder based on Karras' construction algorithm now
working. Fixed an index bug where the wrong prim-bounds were used to
expand leaf node bounding boxes.
