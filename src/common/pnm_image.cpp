// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#if 1//ndef NDEBUG
#include <iostream>
#include <ostream>
#endif

#include <boost/algorithm/string.hpp>

#include "pnm_image.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Helper functions
//

static void load_ascii_rgb(
        uint8_t*        dst,
        std::ifstream&  file,
        size_t          width,
        size_t          height,
        int             /*max_value*/
        )
{
//  assert(max_value < 256);    // TODO: 16-bit
//  assert(max_value == 255);   // TODO: scaling

    for (size_t y = 0; y < height; ++y)
    {
        std::string line;
        std::getline(file, line);

        std::vector<std::string> tokens;
        boost::algorithm::split(
                tokens,
                line,
                boost::algorithm::is_any_of(" \t")
                );

        // Remove empty tokens and  spaces
        tokens.erase(
                std::remove_if(
                        tokens.begin(),
                        tokens.end(),
                        [](std::string str) { return str.empty() || std::isspace(str[0]); }
                        ),
                tokens.end()
                );

        size_t pitch = width * 3;
        assert(tokens.size() == pitch);

        for (size_t x = 0; x < pitch; ++x)
        {
            dst[y * pitch + x] = static_cast<uint8_t>(std::stoi(tokens[x]));
        }
    }
}

static void load_binary_rgb(
        uint8_t*        dst,
        std::ifstream&  file,
        size_t          width,
        size_t          height,
        int             /*max_value*/
        )
{
//  assert(max_value < 256);    // TODO: 16-bit
//  assert(max_value == 255);   // TODO: scaling

    size_t pitch = width * 3;
    file.read(reinterpret_cast<char*>(dst), pitch * height);
}


//-------------------------------------------------------------------------------------------------
// pnm_image
//

bool pnm_image::load(std::string const& filename)
{
    enum format { P1 = 1, P2, P3, P4, P5, P6 };

    std::ifstream file(filename);

    std::string line;

    //
    // First line determines format:
    // P1: ASCII BW         | P4: binary BW
    // P2: ASCII gray scale | P5: binary gray scale
    // P3: ASCII RGB        | P6: binary RGB
    //
    std::getline(file, line);

    if (line.size() < 2 || line[0] != 'P' || line[1] < '0' || line[1] > '6')
    {
        std::cerr << "Invalid file format: " << line << '\n';
        return false;
    }

    format fmt = static_cast<format>(line[1] - '0');

    // Width and height
    for (;;)
    {
        std::getline(file, line);

        if (line[0] == '#')
        {
            // Skip comments
            continue;
        }
        else
        {
            std::vector<std::string> tokens;
            boost::algorithm::split(
                    tokens,
                    line,
                    boost::algorithm::is_any_of(" \t")
                    );

            if (tokens.size() != 2)
            {
                std::cerr << "Invalid dimensions\n";
                return false;
            }

            width_  = static_cast<size_t>(std::stoi(tokens[0]));
            height_ = static_cast<size_t>(std::stoi(tokens[1]));

            break;
        }
    }

    // Max. value
    int max_value = 255;
    if (fmt != P1 && fmt != P4)
    {
        for (;;)
        {
            std::getline(file, line);

            if (line[0] == '#')
            {
                // Skip comments
                continue;
            }
            else
            {
                max_value = std::stoi(line);
                break;
            }
        }
    }

    switch (fmt)
    {
    default:
        std::cerr << "Unsupported PNM image type: P" << std::to_string(static_cast<int>(fmt)) << '\n';
        width_  = 0;
        height_ = 0;
        format_ = PF_UNSPECIFIED;
        data_.resize(0);
        return false;

    case P3:
        assert(max_value < 256);
        assert(max_value == 255);
        format_ = PF_RGB8;
        data_.resize(width_ * height_ * 3);

        load_ascii_rgb(
                data_.data(),
                file,
                width_,
                height_,
                max_value
                );
        return true;

    case P6:
        assert(max_value < 256);
        assert(max_value == 255);
        format_ = PF_RGB8;
        data_.resize(width_ * height_ * 3);

        load_binary_rgb(
                data_.data(),
                file,
                width_,
                height_,
                max_value
                );
        return true;
    }

    return true;
}

}
