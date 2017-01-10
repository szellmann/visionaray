#.rst:
# FindOpenGLES
# --------------
#
# Find the OpenGLES library
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This module defines the following variables:
#
# ::
#
# OPENGLES_INCLUDE_DIR - include directories of OpenGLES
# OPENGLES_LIBRARIES - libraries to link against OpenGLES
# OPENGLES_FOUND - true if OpenGLES has been found and can be used

include(FindPackageHandleStandardArgs)

set(paths
    /opt/vc
)

find_path(OPENGLES_INCLUDE_DIR
    NAMES
        GLES2/gl2.h
    PATHS
        ${paths}
    PATH_SUFFIXES
        include
)

find_library(OPENGLES_LIBRARY
    NAMES
        GLESv2
    PATHS
        ${paths}
    PATH_SUFFIXES
        lib
)

set(OPENGLES_LIBRARIES ${OPENGLES_LIBRARY})

find_package_handle_standard_args(OpenGLES
    DEFAULT_MSG
    OPENGLES_INCLUDE_DIR
    OPENGLES_LIBRARIES
)
