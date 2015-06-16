include(FindPackageHandleStandardArgs)

set(paths
    /usr/
    /usr/local
)

find_path(PTHREAD_INCLUDE_DIR
    NAMES
        pthread.h
    PATH
        ${paths}
    PATH_SUFFIXES
        include
)

find_library(PTHREAD_LIBRARY
    NAMES
        pthread
    PATHS
        ${paths}
    PATH_SUFFIXES
        lib64
        lib32
        lib
)

set(PTHREAD_LIBRARIES ${PTHREAD_LIBRARY})

find_package_handle_standard_args(PTHREAD
    PTHREAD_DEFAULT_MSG
    PTHREAD_INCLUDE_DIR
    PTHREAD_LIBRARY
)
