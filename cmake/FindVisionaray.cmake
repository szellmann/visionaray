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

set(VISIONARAY_LIBRARIES ${VISIONARAY_LIBRARY})

find_package_handle_standard_args(Visionaray
    VISIONARAY_DEFAULT_MSG
    VISIONARAY_INCLUDE_DIR
    VISIONARAY_LIBRARY
)
