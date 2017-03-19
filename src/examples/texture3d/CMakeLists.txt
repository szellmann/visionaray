# This file is distributed under the MIT license.
# See the LICENSE file for details.

find_package(GLUT REQUIRED)

visionaray_use_package(GLUT)

set(EX_TEXTURE3D_SOURCES
    main.cpp
)

visionaray_add_executable(texture3d
    ${EX_TEXTURE3D_SOURCES}
)
