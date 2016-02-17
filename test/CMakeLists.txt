find_package(CUDA)
find_package(Threads)

if (NOT Threads_FOUND)
    message("Threads not found - not building unittests")
    return()
endif()

visionaray_use_package(CUDA)


#--------------------------------------------------------------------------------------------------
# Add googletest as an external project and import its libraries
#

include(ExternalProject)

ExternalProject_Add(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.7.0
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/gtest
    CMAKE_ARGS "-DBUILD_SHARED_LIBS=ON"
    SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/gtest/googletest/googletest
    INSTALL_COMMAND ""
    LOG_DOWNLOAD ON
    LOG_CONFIGURE ON
    LOG_BUILD ON
)

# Gtest's include dir
ExternalProject_Get_Property(googletest SOURCE_DIR)
set(GTEST_INCLUDE_DIR ${SOURCE_DIR}/include)

# Add gtest libraries
ExternalProject_Get_Property(googletest BINARY_DIR)

if(MSVC)
    set(LIBPREFIX "${CMAKE_IMPORT_LIBRARY_PREFIX}")
    set(LIBSUFFIX "${CMAKE_IMPORT_LIBRARY_SUFFIX}")
else()
    set(LIBPREFIX "${CMAKE_SHARED_LIBRARY_PREFIX}")
    set(LIBSUFFIX "${CMAKE_SHARED_LIBRARY_SUFFIX}")
endif()

add_definitions(-DGTEST_LINKED_AS_SHARED_LIBRARY=1)
set(GTEST_LIBRARY_PATH     ${BINARY_DIR}/${CMAKE_CFG_INTDIR}/${LIBPREFIX}gtest${LIBSUFFIX})
set(GTEST_LIBRARYMAIN_PATH ${BINARY_DIR}/${CMAKE_CFG_INTDIR}/${LIBPREFIX}gtest_main${LIBSUFFIX})

set(GTEST_LIBRARY libgtest)
set(GTEST_LIBRARYMAIN libgtest_main)

add_library(${GTEST_LIBRARY} UNKNOWN IMPORTED)
add_library(${GTEST_LIBRARYMAIN} UNKNOWN IMPORTED)

# Set location of gtest libraries
set_property(TARGET ${GTEST_LIBRARY} PROPERTY IMPORTED_LOCATION
                ${GTEST_LIBRARY_PATH} )
set_property(TARGET ${GTEST_LIBRARYMAIN} PROPERTY IMPORTED_LOCATION
                ${GTEST_LIBRARYMAIN_PATH} )

# Make gtest libraries depend on external project
add_dependencies(${GTEST_LIBRARY} googletest)
add_dependencies(${GTEST_LIBRARYMAIN} googletest)


#--------------------------------------------------------------------------------------------------
# Add unittests target
#

# Visionaray include dirs
include_directories(${PROJECT_SOURCE_DIR}/include)


# Unittests executable
set(UNITTESTS_SOURCES
    bvh/traverse.cpp
    math/matrix.cpp
    math/norm.cpp
    math/rect.cpp
    math/simd.cpp
    generic_material.cpp
    render_target.cpp
    swizzle.cpp
    variant.cpp
    version.cpp
)

if(CUDA_FOUND AND VSNRAY_ENABLE_CUDA)
    visionaray_cuda_compile(UNITTESTS_CUDA_SOURCES
#        texture.cu
    )
endif()

# Define executable
visionaray_add_executable(unittests
    ${UNITTESTS_SOURCES}
    ${UNITTESTS_CUDA_SOURCES}
)

target_link_libraries(unittests libgtest libgtest_main ${CMAKE_THREAD_LIBS_INIT})

# Set gtest include dirs as target properties
# This way cmake does not complain about not (yet) existing include dirs
# at first invocation

# Extend already present include dirs list, to retain standard cmake behavior
# (inheritance, etc.)
get_property(ORIGINAL_INCLUDE_DIRS TARGET unittests PROPERTY INCLUDE_DIRECTORIES)

set_target_properties(unittests PROPERTIES
    "INCLUDE_DIRECTORIES" "${ORIGINAL_INCLUDE_DIRS};${GTEST_INCLUDE_DIR}"
)
