## [Unreleased]
### Added
- Quantized 4-wide BVH for CPU. Uncompressed is faster to traverse,
so this is primarily a memory optimization.
- Added support for 4-wide BVHs on the CPU. This currently only
works with index_bvh<>. Uses the SIMD traversal algorithm from Afra
2013, yet (so far) only for 4-wide simd only.

### Changed
- Made pointer_storage and texture_ref trivially constructible.
- Fixed some bugs in the GPU LBVH builder that would cause data
races otherwise.

### Removed
- Support for multi-hit BVH traversal was dropped, in favor of less
complicated traversal routines.
- Support for custom update conditions in trafversal routine was
dropped.

## [0.4.2] - 2024-06-30
### Added
- Experimental support for AMD's HIP GPGPU language. Tested with
the anari-visionaray ANARI device.

### Fixed
- On BVH traversal, hit_record's isect_pos member gets updated.
This is crucial for correctly implementing local shading operations.

## [0.4.1] - 2024-06-05
### Changed
- The Visionaray library is header only now.
- Software texture types and functions can be compiled for device.

## [0.4.0] - 2024-06-05
### Changed
- Getting rid of OpenGL in the core library, moved to common
(apps who use cpu_buffer_rt, pixel_unpack_buffer_rt, etc. now
have to use visionaray::common).
- Optimized BVH traversal on GPU (simpler, and seemingly faster
box test; box test manually inlined into intersect())
- Update build system to adopt modern cmake. Now everything's a
target.

## [0.3.5] - 2024-04-17
### Added
- Compile option VSNRAY_NO_SIMD that the user can set. Then, the
public visionaray headers will not include any gcc intrinsic header
files. Useful in cases where these cause compiler issues.
- Public cuda_texture.h header, to include from CUDA code that
uses the runtime API but is not compiled with nvcc.

### Changed
- Internally use own algos and data structures in favor of thrust.
- CUDA 2D textures can now be reset using a device pointer.

### Removed
- counting_sort functions that were never used.

## [0.3.4] - 2023-12-27
### Added
- Convert from PF_RG8 to PF_RGBA8, to support RG textures.

## [0.3.3] - 2023-12-27
### Added
- Macro CUDA_SAFE_CALL_X() terminates the app if code != cudaSucess.

### Fixed
- When swizzling from PF_Rxx to PF_RGByy, set G and B to 0, not R.
- Fixed a potential division by zero in th Cook-Torrance BRDF
implementation.

## [0.3.2] - 2023-12-21
### Added
- visionarayConfigVersion.cmake to determine the version.

### Fixed
- Accidentally specified the wrong version.

## [0.3.1] - 2023-12-21
### Added
- Version is (also) specified via CMake now.
- Allow CMake install using config scripts.
- Added cylinder as built-in primitive.

### Changed
- Scheduler's scissorBox feature has been replaced with image_region
on the camera.
- Light sample struct has changed, to no longer store the position,
but instead, a direction and distance.
- An accumulation buffer was now added to the builtin render targets
where colors are blended in. For blending kernels, the accumulation
buffer pixel format needs to be specified.

### Fixed
- Fixed an issue where texture coordinates close to an integer were
accidentally truncated to the next lower integer.

## [0.3.0] - 2021-12-25
### Added
- macro VSNRAY_VERSION that can be used in addition to the other
version-related macros.
- Texture storage classes for bricked 3D texture accesses. Those
however didn't prove to be faster than row-major texture storage so
were only added for optional use with the base texture templates.
- "Ray tracing in one weekend" example using CUDA.
- Added an -spp flag to various examples.

### Changed
- Remove VSNRAY_NOT_COPYABLE() macro. Better to use C++11 delete
on copy ctor/assignment operator for this.
- Switched to C++14.
- Significantly overhauled CPU texture implementation.
- Rays now store tmin and tmax, therefore the public interfaces of
functions like any_hit() or is_closer() has changed. It is required
for users who construct rays themselves to fill those values
accordingly.

## [0.2.0] - 2021-02-19
### Added
- Added an -spp flag to viewer that is used by the path tracer.
- Support for HDR/radiance file format in visionaray-common library.
- Added an optional ground plane in viewer.
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
- Pixel sampler types now store parameters indicating ssaa factors or
spp. Those are now evaluated at runtime.
- Don't use clock as LCG seed anymore; rather compute hashes based on
pixelID and frameID.
- BVH instances now store 4x3 instead of 4x4 transform matrices.
- vec3 is no longer aligned to 16 byte boundaries.
- Switched to Ubuntu 16.04 with Travis-CI. We no longer support
Ubuntu 14.04 and gcc 4.8 as that is too old for pbrtParser.
- pbrtParser is now a submodule instead of an external dependency
supplied by the user.

### Removed
- pixel_sampler::ssaa_type<N> was removed and bumped into
pixel_sampler::uniform_type.
- pixel_sampler::jittered_type was removed; the same can be achieved
by using jittered_blend_type and tampering with the blend parameters.

### Fixed
- Various smaller fixes to make things run smoothly on Apple silicon.
- Fixed linking with homebrew GLEW on macOS.

## [0.1.1] - 2020-06-22
### Fixed
- CUDA LBVH builder based on Karras' construction algorithm now
working. Fixed an index bug where the wrong prim-bounds were used to
expand leaf node bounding boxes.
