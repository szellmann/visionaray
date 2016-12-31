#.rst:
# FindVideoCore
# --------------
#
# Find the Raspberry Pi VideoCore APIs
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This module defines the following variables:
#
# ::
#
# VIDEOCORE_INCLUDE_DIR - include directories of VideoCore
# VIDEOCORE_LIBRARIES - libraries to link against VideoCore
# VIDEOCORE_BRCM_GLES2_LIBRARY - Broadcom GLESv2 library
# VIDEOCORE_BRCM_EGL_LIBRARY - Broadcom EGL library
# VIDEOCORE_OPENMAXIL_LIBRARY - OpenMAX IL library
# VIDEOCORE_BCM_HOST_LIBRARY - Broadcom hardware interface library
# VIDEOCORE_VCOS_LIBRARY - vcos library
# VIDEOCORE_VCHIQ_ARM_LIBRARY vchiqarm library
# VIDEOCORE_FOUND - true if VideoCore has been found and can be used

include(FindPackageHandleStandardArgs)

set(paths
    /opt/vc
)

find_path(VIDEOCORE_INCLUDE_DIR
    NAMES
        bcm_host.h
    PATHS
        ${paths}
    PATH_SUFFIXES
        include
)

find_library(VIDEOCORE_BRCM_GLES2_LIBRARY
    NAMES
        brcmGLESv2
    PATHS
        ${paths}
    PATH_SUFFIXES
        lib
)

find_library(VIDEOCORE_BRCM_EGL_LIBRARY
    NAMES
        brcmEGL
    PATHS
        ${paths}
    PATH_SUFFIXES
        lib
)

find_library(VIDEOCORE_OPENMAXIL_LIBRARY
    NAMES
        openmaxil
    PATHS
        ${paths}
    PATH_SUFFIXES
        lib
)


find_library(VIDEOCORE_BCM_HOST_LIBRARY
    NAMES
        bcm_host
    PATHS
        ${paths}
    PATH_SUFFIXES
        lib
)

find_library(VIDEOCORE_VCOS_LIBRARY
    NAMES
        vcos
    PATHS
        ${paths}
    PATH_SUFFIXES
        lib
)


find_library(VIDEOCORE_VCHIQ_ARM_LIBRARY
    NAMES
        vchiq_arm
    PATHS
        ${paths}
    PATH_SUFFIXES
        lib
)

set(VIDEOCORE_LIBRARIES
    ${VIDEOCORE_BRCM_GLES2_LIBRARY}
    ${VIDEOCORE_BRCM_EGL_LIBRARY}
    ${VIDEOCORE_OPENMAXIL_LIBRARY}
    ${VIDEOCORE_BCM_HOST_LIBRARY}
    ${VIDEOCORE_VCOS_LIBRARY}
    ${VIDEOCORE_VCHIQ_ARM_LIBRARY}
)

find_package_handle_standard_args(VideoCore
    DEFAULT_MSG
    VIDEOCORE_INCLUDE_DIR
    VIDEOCORE_LIBRARIES
)
