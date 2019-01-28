// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cstdlib>
#include <iostream>
#include <ostream>
#include <string>
#include <vector>

#include <Support/CmdLine.h>
#include <Support/CmdLineUtil.h>

#include <common/model.h>

using namespace support;
using namespace visionaray;


class converter
{
public:

    void parse_cmd_line(int argc, char** argv);

    std::string input_file;
    std::string output_file;

    file_base::save_options options;

};

void converter::parse_cmd_line(int argc, char** argv)
{
    cl::CmdLine cmd;

    auto ifile = cl::makeOption<std::string&>(
        cl::Parser<>(),
        cmd,
        "input",
        cl::Desc("Input file"),
        cl::Positional,
        cl::Required,
        cl::init(input_file)
        );

    auto ofile = cl::makeOption<std::string&>(
        cl::Parser<>(),
        cmd,
        "output",
        cl::Desc("Output file"),
        cl::Positional,
        cl::Required,
        cl::init(output_file)
        );


    auto args = std::vector<std::string>(argv + 1, argv + argc);
    cl::expandWildcards(args);
    cl::expandResponseFiles(args, cl::TokenizeUnix());

    try
    {
        cmd.parse(args);
    }
    catch (...)
    {
        std::cout << cmd.help(argv[0]) << '\n';
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv)
{
    converter conv;
    conv.parse_cmd_line(argc, argv);

    model mod;
    if (!mod.load(conv.input_file))
    {
        std::cout << "Error: cannot load input file: " << conv.input_file << '\n';
        exit(EXIT_FAILURE);
    }

    if (!mod.save(conv.output_file, conv.options))
    {
        std::cout << "Error: cannot save output file: " << conv.output_file << '\n';
        exit(EXIT_FAILURE);
    }
}
