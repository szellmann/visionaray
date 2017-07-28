// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cstdint>
#include <fstream>
#include <iostream>
#include <ostream>

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

#include <visionaray/math/vector.h>

#include "hdr_image.h"

namespace visionaray
{

bool hdr_image::load(std::string const& filename)
{
    std::ifstream file(filename);

    if (!file.good())
    {
        std::cerr << "Cannot open HDR file: " << filename << '\n';
        return false;
    }

    // parse information header ---------------------------

    std::string line;

    std::getline(file, line);

    if (line != "#?RADIANCE")
    {
        std::cerr << "Invalid Radiance picture file\n";
        return false;
    }

    enum cformat { RGBE, XYZE };
    cformat format = RGBE;

    while (!line.empty())
    {
        std::getline(file, line);

        std::vector<std::string> vars;
        boost::split(vars, line, boost::is_any_of("="));

        if (vars.size() >= 2)
        {
            std::string key = vars[0];
            boost::trim(vars[0]);
            vars.erase(vars.begin());
            std::string value = boost::algorithm::join(vars, "");

            if (key == "FORMAT" && value == "32-bit_rle_rgbe")
            {
                format = RGBE;
            }
            else if (key == "FORMAT" && value == "32-bit_rle_xyze")
            {
                format = XYZE;
                std::cerr << "Error: XYZE format in HDR files not yet supported\n";
                return false;
            }
            else if (key == "FORMAT")
            {
                std::cerr << "Error: unsupported format string in HDR file\n";
                return false;
            }
        }
    }

    // resolution string ----------------------------------

    std::getline(file, line);
    std::cout << "Line: " << line << std::endl;

    std::vector<std::string> res;
    boost::split(res, line, boost::is_any_of("\t "));

    if (res.size() != 4)
    {
        std::cerr << "Error: invalid resolution string in HDR file\n";
        return false;
    }

    if (res[0] == "-Y" && res[2] == "+X")
    {
        width_  = boost::lexical_cast<size_t>(res[3]);
        height_ = boost::lexical_cast<size_t>(res[1]);
    }
    else
    {
        std::cerr << "Error: unsupported resolution string in HDR file\n";
        return false;
    }

    // scanlines ------------------------------------------

    std::vector<vec4> rgbe_line(width_);

    for (size_t h = 0; h < height_; ++h)
    {
        std::getline(file, line);
        auto bytes = reinterpret_cast<uint8_t const*>(line.c_str());

        // Read the first four bytes of each scaline
        // Radiance's new RLE: 0x2 0x2 means RLE
        uint8_t r = *bytes++;
        uint8_t g = *bytes++;
        uint8_t b = *bytes++;
        uint8_t e = *bytes++;

        if (r == 2 && g == 2)
        {
            // Component-wise RLE
            for (size_t c = 0; c < 4; ++c)
            {
                size_t x = 0;
                while (x < width_)
                {
                    uint8_t runlength = *bytes++;

                    if (runlength <= 128) // no run!
                    {
                        for (size_t i = 0; i < runlength; ++i)
                        {
                            rgbe_line[i][c] = *bytes++;
                        }
                    }
                    else
                    {
                        runlength -= 128;

                        for (size_t i = 0; i < runlength; ++i)
                        {
                        }
                    }

                    x += runlength;
                }
                std::cout << x << '\n';
            }
        }
    }

    return true;
}

} // visionaray
