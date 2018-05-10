#.rst:
# FindXCB
# --------------
#
# Find the X protocol C-language Binding (XCB) library
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This module defines the following variables:
#
# ::
#
# XCB_INCLUDE_DIR - include directories of XCB
# XCB_LIBRARIES - libraries to link against XCB
# XCB_FOUND - true if XCB has been found and can be used

include(FindPackageHandleStandardArgs)

set(paths
    /usr
    /usr/local
    /opt/X11
)

find_path(XCB_INCLUDE_DIR
    NAMES
        xcb/xcb.h
    PATHS
        ${paths}
    PATH_SUFFIXES
        include
)

find_library(XCB_LIBRARY
    NAMES
        xcb
    PATHS
        ${paths}
    PATH_SUFFIXES
        lib64
        lib
)

find_library(X11_XCB_LIBRARY
    NAMES
        X11-xcb
    PATHS
        ${paths}
    PATH_SUFFIXES
        lib64
        lib
)

set(XCB_INCLUDE_DIRS ${XCB_INCLUDE_DIR})
set(XCB_LIBRARIES ${XCB_LIBRARY} ${X11_XCB_LIBRARY})

find_package_handle_standard_args(XCB
    DEFAULT_MSG
    XCB_INCLUDE_DIR
    XCB_LIBRARY
)
