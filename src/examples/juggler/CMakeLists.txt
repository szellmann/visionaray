# This file is distributed under the MIT license.
# See the LICENSE file for details.

find_package(GLUT REQUIRED)

visionaray_use_package(GLUT)

set(EX_JUGGLER_SOURCES
    main.cpp
)

visionaray_add_executable(juggler
    ${EX_JUGGLER_SOURCES}
)
