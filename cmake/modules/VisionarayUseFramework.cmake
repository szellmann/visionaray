#---------------------------------------------------------------------------------------------------
# visionaray_use_framework(name)
#


function(visionaray_use_framework name)
    find_library(FRAMEWORK_${name}
        NAMES
            ${name}
        PATHS
            ${CMAKE_OSX_SYSROOT}/System/Library
        PATH_SUFFIXES
            Frameworks
        NO_DEFAULT_PATH
    )

    if(${FRAMEWORK_${name}} STREQUAL FRAMEWORK_${name}-NOTFOUND)
        message(ERROR ": Framework ${name} not found")
    else()
        set(__VSNRAY_LINK_LIBRARIES ${__VSNRAY_LINK_LIBRARIES} ${FRAMEWORK_${name}}/${name} PARENT_SCOPE)
    endif()
endfunction()
