// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include "Support/CmdLine.h"
#include "Support/CmdLineUtil.h"

#include "PrettyPrint.h"

#include <list>

#include <gtest/gtest.h>

using namespace support;

typedef std::vector<std::string> Argv;

bool parse(cl::CmdLine& cmd, Argv const& argv)
{
    try
    {
        cmd.parse(argv);
        return true;
    }
    catch (std::exception& e)
    {
        std::cout << "ERROR: " << e.what() << std::endl;
        return false;
    }
}

static void GenerateFiles()
{
    std::ofstream rsp1("CmdLineExpandTest.rsp1");
    rsp1 << "@CmdLineExpandTest.rsp2 -f1 111 -f2222 -f3=333";

    std::ofstream rsp2("CmdLineExpandTest.rsp2");
    rsp2 << "-g1 111 -g2222 -g3=333";

    std::vector<std::string> files = {
        "file1.txt", "file2.txt", "file3.txt",
        "file11.txt", "file12.txt", "file13.txt",
    };

    for (auto&& f : files) {
        std::ofstream file(f);
        file << "Hello";
    }
}

TEST(CmdLineExpand, Test1)
{
    GenerateFiles();

    cl::CmdLine cmd;

    auto f1 = cl::makeOption<int>(cl::Parser<>(), cmd, "f1", cl::ArgRequired);
    auto f2 = cl::makeOption<int>(cl::Parser<>(), cmd, "f2", cl::Prefix);
    auto f3 = cl::makeOption<int>(cl::Parser<>(), cmd, "f3", cl::ArgOptional);
    auto g1 = cl::makeOption<int>(cl::Parser<>(), cmd, "g1", cl::ArgRequired);
    auto g2 = cl::makeOption<int>(cl::Parser<>(), cmd, "g2", cl::Prefix);
    auto g3 = cl::makeOption<int>(cl::Parser<>(), cmd, "g3", cl::ArgOptional);
    auto h1 = cl::makeOption<int>(cl::Parser<>(), cmd, "h1", cl::ArgRequired);
    auto h2 = cl::makeOption<int>(cl::Parser<>(), cmd, "h2", cl::Prefix);
    auto h3 = cl::makeOption<int>(cl::Parser<>(), cmd, "h3", cl::ArgOptional);

    Argv args = { "-h1", "111", "@CmdLineExpandTest.rsp1", "-h2222", "-h3=333" };

    cl::expandResponseFiles(args, cl::TokenizeUnix());

    std::cout << "command-line: " << pretty(args) << std::endl;

    EXPECT_TRUE( parse(cmd, args) );
}

#ifdef _MSC_VER

// VC only: open ifstream using wstring's
TEST(CmdLineExpand, TestWchar)
{
    //GenerateFiles();

    std::list<std::wstring> args = { L"-h1", L"111", L"@CmdLineExpandTest.rsp1", L"-h2222", L"-h3=333" };

    cl::expandResponseFiles(args, cl::TokenizeWindows());

    std::wcout << L"command-line: " << pretty(args) << std::endl;
}

#endif

#ifdef _WIN32

TEST(CmdLineExpand, TestWildcards)
{
    //GenerateFiles();

    {
        std::vector<std::string> args = {"kjhasfjkhasdf"};
        support::cl::expandWildcards(args);
        EXPECT_EQ(1, args.size());
        std::cout << pretty(args) << std::endl;
    }
    {
        std::vector<std::string> args = {"file*.t?t"};
        support::cl::expandWildcards(args);
        EXPECT_EQ(6, args.size());
        std::cout << pretty(args) << std::endl;
    }
    {
        std::vector<std::string> args = {"file?.txt"};
        support::cl::expandWildcards(args);
        EXPECT_EQ(3, args.size());
        std::cout << pretty(args) << std::endl;
    }
    {
        std::vector<std::string> args = {"file2?.txt", "file1?.txt"};
        support::cl::expandWildcards(args);
        EXPECT_EQ(5, args.size());
        std::cout << pretty(args) << std::endl;
    }
    {
        std::vector<std::string> args = {"file1?.txt", "file2?.txt", "file5?.txt"};
        support::cl::expandWildcards(args);
        EXPECT_EQ(6, args.size());
        std::cout << pretty(args) << std::endl;
    }
}

#endif
