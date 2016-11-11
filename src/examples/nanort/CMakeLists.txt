find_package(GLUT REQUIRED)

visionaray_use_package(GLUT)

set(NANORT_DIR ${PROJECT_SOURCE_DIR}/src/3rdparty/nanort)
set(NANORT_INCLUDE_DIR ${NANORT_DIR})

include_directories(${NANORT_INCLUDE_DIR})

set(EX_NANORT_SOURCES
    main.cpp
)

visionaray_add_executable(nanort
    ${EX_NANORT_SOURCES}
)
