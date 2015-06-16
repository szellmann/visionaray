if(VSNRAY_ENABLE_COVER)
add_subdirectory(cover)
endif()

if(VSNRAY_ENABLE_EXAMPLES)
add_subdirectory(examples)
endif()

if(VSNRAY_ENABLE_VIEWER)
add_subdirectory(viewer)
endif()

add_subdirectory(visionaray)
