# This file is distributed under the MIT license.
# See the LICENSE file for details.


#---------------------------------------------------------------------------------------------------
# Support for AMD ROCm HCC compiler
#

if(UNIX)

    # Determine if we compile w/ hcc by checking ${CXX} --version

    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} --version
        OUTPUT_VARIABLE output
        )

    if("${output}" MATCHES ".*HCC.*")

        # invoke hcc-config

        set(__VSNRAY_CXX_COMPILER_IS_HCC ON)

        get_filename_component(
            hcc_base_path ${CMAKE_CXX_COMPILER} DIRECTORY
            )

        set(hcc_config_cmd "${hcc_base_path}/hcc-config")

        execute_process(
            COMMAND ${hcc_config_cmd} --cxxflags
            OUTPUT_VARIABLE cxxflags
            )

        string(REPLACE "-I" "-isystem " cxxflags "${cxxflags}")

        set(__VSNRAY_HCC_COMPILE_FLAGS "${cxxflags}")

        execute_process(
            COMMAND ${hcc_config_cmd} --ldflags
            OUTPUT_VARIABLE ldflags
            )

        set(__VSNRAY_HCC_LINK_FLAGS "${ldflags} -lhc_am -lm")
    endif()

endif()
