if(VSNRAY_ENABLE_COVER)
add_subdirectory(cover)
endif()

if(VSNRAY_ENABLE_EXAMPLES)
add_subdirectory(examples)
endif()

add_subdirectory(viewer)
add_subdirectory(visionaray)
