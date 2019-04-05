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
        get_filename_component(suffix ${f} EXT)

        if(NOT ${suffix} STREQUAL ".cu")
            message(FATAL_ERROR "Cannot cuda_compile file with extension ${suffix}")
            return()
        endif()

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
endfunction()

function(visionaray_add_library name)
    add_library(${name} ${ARGN})
    target_link_libraries(${name} ${__VSNRAY_LINK_LIBRARIES})
endfunction()
