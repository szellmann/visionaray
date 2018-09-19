#.rst:
# FindPtex
# --------------
#
# Find Disney's Ptex library
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This module defines the following variables:
#
# ::
#
# PTEX_INCLUDE_DIR - include directories of Ptex
# PTEX_LIBRARIES - libraries to link against Ptex
# PTEX_FOUND - true if Ptex has been found and can be used

include(FindPackageHandleStandardArgs)

set(paths
    /usr
    /usr/local
)

find_path(PTEX_INCLUDE_DIR
    NAMES
        Ptexture.h
    PATHS
        ${paths}
    PATH_SUFFIXES
        include
)

find_library(PTEX_LIBRARY
    NAMES
        Ptex
    PATHS
        ${paths}
    PATH_SUFFIXES
        lib64
        lib
)

set(PTEX_INCLUDE_DIRS ${PTEX_INCLUDE_DIR})
set(PTEX_LIBRARIES ${PTEX_LIBRARY})

find_package_handle_standard_args(PTEX
    DEFAULT_MSG
    PTEX_INCLUDE_DIR
    PTEX_LIBRARY
)
