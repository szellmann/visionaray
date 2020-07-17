# This file is distributed under the MIT license.
# See the LICENSE file for details.

macro(visionaray_link_libraries)
    set(__VSNRAY_LINK_LIBRARIES ${__VSNRAY_LINK_LIBRARIES} ${ARGN})
endmacro()

function(visionaray_target_set_warnings target)
    if(MSVC)

    elseif(__COMPILER_PGI)

    else() # GNU, Clang, etc.

        #------------------------------------------------------
        # Enable warnings
        #

        target_compile_options(
            ${target} PRIVATE
            -Wmissing-braces
            -Wsign-compare
            -Wwrite-strings
            -Woverloaded-virtual
            -Wundef
        )

        if(VSNRAY_ENABLE_WARNINGS)
            target_compile_options(${target} PRIVATE -Wall -Wextra)

            if(VSNRAY_ENABLE_PEDANTIC)
                target_compile_options(${target} PRIVATE -pedantic)
            endif()
        endif()


        #------------------------------------------------------
        # Selectively disable warnings
        #

        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 6.0)
                # Disable warnings like:
                # warning: ignoring attributes on template argument __m128i
                target_compile_options(${target} PUBLIC -Wno-ignored-attributes)
            endif()
        endif()
    endif()
endfunction()

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
    visionaray_target_set_warnings(${name})
    target_link_libraries(${name} ${__VSNRAY_LINK_LIBRARIES})

    if(VSNRAY_MACOSX_BUNDLE)
        set_target_properties(${name} PROPERTIES MACOSX_BUNDLE TRUE)
    endif()
endfunction()

function(visionaray_add_cuda_executable name)
    if(NOT CUDA_FOUND OR NOT VSNRAY_ENABLE_CUDA)
        return()
    endif()

    cuda_add_executable(${name} ${ARGN})
    #visionaray_target_set_warnings(${name})
    target_link_libraries(${name} ${__VSNRAY_LINK_LIBRARIES})

    if(VSNRAY_MACOSX_BUNDLE)
        set_target_properties(${name} PROPERTIES MACOSX_BUNDLE TRUE)
    endif()
endfunction()

function(visionaray_add_library name)
    add_library(${name} ${ARGN})
    visionaray_target_set_warnings(${name})
    target_link_libraries(${name} ${__VSNRAY_LINK_LIBRARIES})
endfunction()
