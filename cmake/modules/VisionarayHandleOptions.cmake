#---------------------------------------------------------------------------------------------------
# Module to handle VSNRAY_* options
#

if(MSVC)

else()

    add_definitions(-Wmissing-braces)
    add_definitions(-Wsign-compare)
    add_definitions(-Wwrite-strings)
    add_definitions(-Woverloaded-virtual)

    if(VSNRAY_ENABLE_WARNINGS)

        add_definitions(-Wall -Wextra)

        if(VSNRAY_ENABLE_PEDANTIC)
            add_definitions(-pedantic)
        endif()

    endif()
endif()

