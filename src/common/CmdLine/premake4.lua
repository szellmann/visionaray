----------------------------------------------------------------------------------------------------
if _ACTION == "clean" then
    os.rmdir("build")
end

----------------------------------------------------------------------------------------------------
local have_gtest = os.isdir("test/gtest")

----------------------------------------------------------------------------------------------------
solution "Support"

    configurations { "Release", "Debug" }

    platforms { "x64", "x32" }

    location    ("build/" .. _ACTION)
    objdir      ("build/" .. _ACTION .. "/obj")

    if have_gtest then
        -- all tests are single threaded.
        -- compiling with "-pthread" doesn't help... avoid linker errors...
        -- FIXME!
        defines { "GTEST_HAS_PTHREAD=0" }
    end

    configuration { "Debug" }
        defines { "_DEBUG" }
        flags { "ExtraWarnings", "Symbols" }

    configuration { "Release" }
        defines { "NDEBUG" }
        flags { "ExtraWarnings", "Optimize" }

    configuration { "Debug", "x64" }
        targetdir ("build/" .. _ACTION .. "/bin/x64/Debug")

    configuration { "Release", "x64" }
        targetdir ("build/" .. _ACTION .. "/bin/x64/Release")

    configuration { "Debug", "x32" }
        targetdir ("build/" .. _ACTION .. "/bin/x32/Debug")

    configuration { "Release", "x32" }
        targetdir ("build/" .. _ACTION .. "/bin/x32/Release")

    configuration { "gmake" }
        buildoptions {
            "-std=c++11",
            "-pedantic",
        }

    configuration { "windows" }
        flags { "Unicode" }

----------------------------------------------------------------------------------------------------
project "CmdLine"

    kind "StaticLib"

    language "C++"

    includedirs { "include/" }

    files {
        "include/**.*",
        "src/**.*",
    }

    configuration { "Debug" }
        targetsuffix "d"

----------------------------------------------------------------------------------------------------
project "Test"

    kind "ConsoleApp"

    language "C++"

    links { "CmdLine" }

    includedirs { "include/" }

    files {
        "test/*.cpp",
        "test/*.h",
    }

----------------------------------------------------------------------------------------------------
function add_unittest(name)
    project (name)

        kind "ConsoleApp"

        language "C++"

        links { "CmdLine", "gtest", "gtest_main" }

        includedirs { "include/", "test/" }

        if have_gtest then
            includedirs { "test/gtest/include" }
        end

        files { "test/unittests/" .. name .. ".cpp" }
end

add_unittest("CmdLineExpandTest")
add_unittest("CmdLineTest")
add_unittest("CmdLineToArgvTest")
add_unittest("StringRefTest")
add_unittest("StringSplitTest")

----------------------------------------------------------------------------------------------------
if have_gtest then

    project "gtest"

        kind "StaticLib"

        language "C++"

        includedirs {
            "test/gtest/include",
            "test/gtest",
        }

        files {
            "test/gtest/src/gtest-all.cc",
        }

    project "gtest_main"

        kind "StaticLib"

        language "C++"

        includedirs {
            "test/gtest/include",
            "test/gtest",
        }

        files {
            "test/gtest/src/gtest_main.cc",
        }

end
