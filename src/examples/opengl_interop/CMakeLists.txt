# This file is distributed under the MIT license.
# See the LICENSE file for details.

enable_language(CUDA)
add_executable(opengl_interop)
target_sources(opengl_interop PRIVATE main.cu)
target_link_libraries(opengl_interop PUBLIC visionaray visionaray_common)
target_compile_definitions(opengl_interop PUBLIC APPDIR="${CMAKE_CURRENT_SOURCE_DIR}")
