#.rst:
# FindVisionaray
# --------------
#
# Find the Visionaray ray tracing library
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This module defines the following variables:
#
# ::
#
# VISIONARAY_INCLUDE_DIR - include directories of Visionaray
# VISIONARAY_LIBRARIES - libraries to link against Visionaray
# VISIONARAY_FOUND - true if Visionaray has been found and can be used

include(FindPackageHandleStandardArgs)

set(paths
    /usr
    /usr/local
)

find_path(VISIONARAY_INCLUDE_DIR
    NAMES
        scheduler.h # TODO: sure?
    PATHS
        ${paths}
    PATH_SUFFIXES
        include
        include/visionaray
)

find_library(VISIONARAY_LIBRARY
    NAMES
        visionaray
    PATHS
        ${paths}
    PATH_SUFFIXES
        lib64
        lib
)

find_library(VISIONARAY_COMMON_LIBRARY
    NAMES
        visionaray_common
    PATHS
        ${paths}
    PATH_SUFFIXES
        lib64
        lib
)

set(VISIONARAY_INCLUDE_DIRS ${VISIONARAY_INCLUDE_DIR})
set(VISIONARAY_LIBRARIES ${VISIONARAY_LIBRARY} ${VISIONARAY_COMMON_LIBRARY})

find_package_handle_standard_args(Visionaray
    DEFAULT_MSG
    VISIONARAY_INCLUDE_DIR
    VISIONARAY_LIBRARY
)
