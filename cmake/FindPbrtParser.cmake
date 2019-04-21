#.rst:
# FindPtex
# --------------
#
# Find Ingo Wald's pbrt-parser library
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This module defines the following variables:
#
# ::
#
# PBRTPARSER_INCLUDE_DIR - include directories of pbrt-parser
# PBRTPARSER_LIBRARIES - libraries to link against pbrt-parser
# PBRTPARSER_FOUND - true if pbrt-parser has been found and can be used

include(FindPackageHandleStandardArgs)

set(paths
    /usr
    /usr/local
)

find_path(PBRTPARSER_INCLUDE_DIR
    NAMES
        Scene.h
    PATHS
        ${paths}
    PATH_SUFFIXES
        include/
        include/pbrtParser
)

find_library(PBRTPARSER_LIBRARY
    NAMES
        pbrtParser
    PATHS
        ${paths}
    PATH_SUFFIXES
        lib64
        lib
)

set(PBRTPARSER_INCLUDE_DIRS ${PBRTPARSER_INCLUDE_DIR})
set(PBRTPARSER_LIBRARIES ${PBRTPARSER_LIBRARY})

find_package_handle_standard_args(PBRTPARSER
    DEFAULT_MSG
    PBRTPARSER_INCLUDE_DIR
    PBRTPARSER_LIBRARY
)
