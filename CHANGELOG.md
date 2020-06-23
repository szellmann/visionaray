## [Unreleased]
### Changed
- pbrtParser is now a submodule instead of an external dependency
supplied by the user.

## [0.1.1] - 2020-06-22
### Fixed
- CUDA LBVH builder based on Karras' construction algorithm now
working. Fixed an index bug where the wrong prim-bounds were used to
expand leaf node bounding boxes.
