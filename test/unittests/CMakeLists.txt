find_package(Boost COMPONENTS chrono filesystem iostreams system thread REQUIRED)
find_package(CUDA)
find_package(Git)
find_package(Threads)

if (NOT GIT_FOUND)
    message("Git not found - not building unittests")
    return()
endif()

if (NOT Threads_FOUND)
    message("Threads not found - not building unittests")
    return()
endif()

visionaray_use_package(Boost)
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
    bvh/build.cpp
#   bvh/traverse.cpp
    detail/algorithm.cpp
    math/simd/gather.cpp
    math/simd/simd.cpp
    math/matrix.cpp
    math/ray.cpp
    math/rectangle.cpp
    math/snorm.cpp
    math/unorm.cpp
    math/vector.cpp
    array.cpp
    generic_material.cpp
    generic_primitive.cpp
    get_normal.cpp
    render_target.cpp
    sampling.cpp
    swizzle.cpp
    variant.cpp
    version.cpp
)

if(CUDA_FOUND AND VSNRAY_ENABLE_CUDA)
    cuda_include_directories(${GTEST_INCLUDE_DIR})
    visionaray_cuda_compile(UNITTESTS_CUDA_SOURCES
        cuda/cast.cu
        array.cu
        texture.cu
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

# Allow multi-line comments, some unit tests have verbose documentation with extra '/'s and '\'s
if(CMAKE_COMPILER_IS_GNUCXX)
    get_property(ORIGINAL_COMPILE_FLAGS TARGET unittests PROPERTY COMPILE_FLAGS)
    set_target_properties(unittests PROPERTIES
        "COMPILE_FLAGS" "${ORIGINAL_COMPILE_FLAGS} -Wno-comment"
    )
endif()


add_test(unittests unittests)
