# This file is distributed under the MIT license.
# See the LICENSE file for details.

macro(visionaray_link_libraries)
    set(__VSNRAY_LINK_LIBRARIES ${__VSNRAY_LINK_LIBRARIES} ${ARGN})
endmacro()

function(visionaray_cuda_compile outfiles)
    if(NOT CUDA_FOUND OR NOT VSNRAY_ENABLE_CUDA)
        return()
    endif()

    foreach(f ${ARGN})
        if(BUILD_SHARED_LIBS)
            cuda_compile(cuda_compile_obj ${f} SHARED)
        else()
            cuda_compile(cuda_compile_obj ${f})
        endif()
        set(out ${out} ${f} ${cuda_compile_obj})
    endforeach()

    set(${outfiles} ${out} PARENT_SCOPE)
endfunction()

function(visionaray_add_executable name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} ${__VSNRAY_LINK_LIBRARIES})

    if(__VSNRAY_CXX_COMPILER_IS_HCC)
        set_target_properties(${name} PROPERTIES
            COMPILE_FLAGS  ${__VSNRAY_HCC_COMPILE_FLAGS}
            LINK_FLAGS ${__VSNRAY_HCC_LINK_FLAGS}
        )
    endif()
endfunction()

function(visionaray_add_library name)
    add_library(${name} ${ARGN})
    target_link_libraries(${name} ${__VSNRAY_LINK_LIBRARIES})

    if(__VSNRAY_CXX_COMPILER_IS_HCC)
        set_target_properties(${name} PROPERTIES
            COMPILE_FLAGS  ${__VSNRAY_HCC_COMPILE_FLAGS}
            LINK_FLAGS ${__VSNRAY_HCC_LINK_FLAGS}
        )
    endif()
endfunction()
