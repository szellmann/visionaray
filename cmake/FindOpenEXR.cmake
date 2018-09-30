#.rst:
# FindOpenEXR
# --------------
#
# Find OpenEXR library
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This module defines the following variables:
#
# ::
#
# OPENEXR_INCLUDE_DIR - include directories of OpenEXR
# OPENEXR_LIBRARIES - libraries to link against OpenEXR
# OPENEXR_FOUND - true if OpenEXR has been found and can be used

include(FindPackageHandleStandardArgs)

set(paths
    /usr
    /usr/local
)

find_path(OPENEXR_INCLUDE_DIR
    NAMES
        OpenEXRConfig.h
    PATHS
        ${paths}
    PATH_SUFFIXES
        include
        include/openexr
        include/OpenEXR
)

set(openexr_libraries Half Iex IlmImf IlmThread Imath)

foreach (lib ${openexr_libraries})
    string(TOUPPER ${lib} lib_upper)
    find_library(OPENEXR_${lib_upper}_LIBRARY
        NAMES
            ${lib}
        PATHS
            ${paths}
        PATH_SUFFIXES
            lib64
            lib
    )
endforeach()


set(OPENEXR_INCLUDE_DIRS ${OPENEXR_INCLUDE_DIR})
set(OPENEXR_LIBRARIES
    ${OPENEXR_ILMTHREAD_LIBRARY}
    ${OPENEXR_ILMIMF_LIBRARY}
    ${OPENEXR_IMATH_LIBRARY}
    ${OPENEXR_IEX_LIBRARY}
    ${OPENEXR_HALF_LIBRARY}
)

find_package_handle_standard_args(OpenEXR
    DEFAULT_MSG
    OPENEXR_INCLUDE_DIR
    OPENEXR_LIBRARIES
)
