// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include "Support/CmdLineUtil.h"

#include "PrettyPrint.h"

#include <vector>
#include <iostream>

#include <gtest/gtest.h>
#ifdef _WIN32
#include <windows.h>
#endif

using namespace support;

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
#ifdef _WIN32

static std::string toUTF8(std::wstring const& str)
{
    // Get the length of the UTF-8 encoded string
    int len = ::WideCharToMultiByte(
        CP_UTF8, 0, str.data(), static_cast<int>(str.size()), NULL, 0, NULL, NULL);

    if (len == 0)
    {
    }

    std::vector<char> buf(len);

    // Convert from UTF-16 to UTF-8
    len = ::WideCharToMultiByte(
        CP_UTF8, 0,
        str.data(), static_cast<int>(str.size()),
        buf.data(), static_cast<int>(buf.size()),
        NULL, NULL);

    if (len == 0)
    {
    }

    return std::string(buf.begin(), buf.end());
}

static std::wstring toUTF16(std::string const& str)
{
    // Get the length of the UTF-16 encoded string
    int len = ::MultiByteToWideChar(
        CP_UTF8, MB_ERR_INVALID_CHARS, str.data(), static_cast<int>(str.size()), NULL, 0);

    if (len == 0)
    {
    }

    std::vector<wchar_t> buf(len);

    // Convert from UTF-8 to UTF-16
    len = ::MultiByteToWideChar(
        CP_UTF8, MB_ERR_INVALID_CHARS,
        str.data(), static_cast<int>(str.size()),
        buf.data(), static_cast<int>(buf.size()));

    if (len == 0)
    {
    }

    return std::wstring(buf.begin(), buf.end());
}

static std::vector<std::string> stringToArgvWindows(std::wstring const& wargs)
{
    int argc = 0;
    auto argv = ::CommandLineToArgvW(wargs.c_str(), &argc);

    std::vector<std::string> result;

    for (int i = 0; i < argc; ++i)
    {
        result.push_back(toUTF8(argv[i]));
    }

    LocalFree(argv);

    return result;
}

static std::vector<std::string> stringToArgvCL(std::string const& args)
{
    std::vector<std::string> argv;

    cl::tokenizeCommandLineWindows(args.begin(), args.end(), std::back_inserter(argv));

    return argv;
}

static std::string quoteArgs(std::vector<std::string> const& argv)
{
    std::string args;

    cl::quoteArgsWindows(argv.begin(), argv.end(), std::back_inserter(args));

    return args;
}

static void compare(std::string const& args)
{
    SCOPED_TRACE("command-line: R\"(" + args + ")\"");

    auto x = stringToArgvWindows(toUTF16(args));
    auto y = stringToArgvCL(args);

    EXPECT_EQ(x, y);

    auto s = quoteArgs(y);
    auto z = stringToArgvCL(s);

    SCOPED_TRACE("quoted: R\"(" + s + ")\"");

    EXPECT_EQ(x, z);
}

TEST(CmdLineToArgvTest, Win)
{
//  #define P R"("c:\path to\test program.exe" )"
//  #define P R"("c:\path to\test program.exe )"
    #define P R"(test )"

    const std::string tests[] = {
        P R"()",
        P R"( )",
        P R"( ")",
        P R"( " )",
        P R"(foo""""""""""""bar)",
        P R"(foo"X""X""X""X"bar)",
        P R"(   "this is a string")",
        P R"("  "this is a string")",
        P R"( " "this is a string")",
        P R"("this is a string")",
        P R"("this is a string)",
        P R"("""this" is a string)",
        P R"("hello\"there")",
        P R"("hello\\")",
        P R"(abc)",
        P R"("a b c")",
        P R"(a"b c d"e)",
        P R"(a\"b c)",
        P R"("a\"b c")",
        P R"("a b c\\")",
        P R"("a\\\"b")",
        P R"(a\\\b)",
        P R"("a\\\b")",
        P R"("\"a b c\"")",
        P R"("a b c" d e)",
        P R"("ab\"c" "\\" d)",
        P R"(a\\\b d"e f"g h)",
        P R"(a\\\"b c d)",
        P R"(a\\\\"b c" d e)",
        P R"("a b c""   )",
        P R"("""a""" b c)",
        P R"("""a b c""")",
        P R"(""""a b c"" d e)",
        P R"("c:\file x.txt")",
        P R"("c:\dir x\\")",
        P R"("\"c:\dir x\\\"")",
        P R"("a b c")",
        P R"("a b c"")",
        P R"("a b c""")",
        P R"(a b c)",
        P R"(a\tb c)",
        P R"(a\nb c)",
        P R"(a\vb c)",
        P R"(a\fb c)",
        P R"(a\rb c)",
    };

    for (auto const& s : tests)
    {
        EXPECT_NO_FATAL_FAILURE( compare(s) );
    }
}

#endif

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
TEST(CmdLineToArgvTest, Unix)
{
    // FIXME:
    // Add some tests here...
}
