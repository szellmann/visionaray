// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <ostream>

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

#include "hdr_image.h"

namespace visionaray
{

bool hdr_image::load(std::string const& filename)
{
    std::ifstream file(filename);

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

    std::vector<std::string> res;
    boost::split(res, line, boost::is_any_of("\t "));

    if (res.size() != 4)
    {
        std::cerr << "Error: invalid resolution string in HDR file\n";
    }

    if (res[0] == "-Y" && res[2] == "+X")
    {
        width_  = boost::lexical_cast<size_t>(res[3]);
        height_ = boost::lexical_cast<size_t>(res[1]);
    }
    else
    {
        std::cerr << "Error: unsupported resolution string in HDR file\n";
    }

    format_ = PF_RGB32F;

    data_.resize(width_ * height_ * 3 * sizeof(float));

    float* data = reinterpret_cast<float*>(data_.data());

    // scanlines ------------------------------------------

    for (size_t y = 0; y < height_; ++y)
    {
        // Read the scanline header
        // two bytes equal 2 indicate new format
        // followed by upper and lower byte of
        // the scanline length (< 32768)
        uint8_t header[4];
        file.read((char*)&header, sizeof(header));

        if (header[0] == 2 && header[1] == 2)
        {
            unsigned len = (header[2] << 8) | header[3];

            using RGBE = std::array<uint8_t, 4>;
            std::vector<RGBE> rgbe(len);

            for (unsigned c = 0; c < 4; ++c)
            {
                uint8_t rl = 0;
                uint8_t bytes[128];
                unsigned x = 0;

                while (x < len)
                {
                    file.read((char*)&rl, sizeof(rl));

                    unsigned num_pixels = 0;

                    if (rl > 128)
                    {
                        // This is a run
                        num_pixels = rl & 127;
                        file.read((char*)bytes, 1);

                        for (unsigned i = 0; i < num_pixels; ++i)
                        {
                            rgbe[x + i][c] = bytes[0];
                        }
                    }
                    else
                    {
                        num_pixels = rl;
                        file.read((char*)bytes, num_pixels);

                        for (unsigned i = 0; i < num_pixels; ++i)
                        {
                            rgbe[x + i][c] = bytes[i];
                        }
                    }

                    x += num_pixels;
                }
            }

            for (unsigned x = 0; x < len; ++x)
            {
                float e = rgbe[x][3] - 128;
                float be = powf(2.0f, e);
                *data++ = (rgbe[x][0] / 256.0f) * be;
                *data++ = (rgbe[x][1] / 256.0f) * be;
                *data++ = (rgbe[x][2] / 256.0f) * be;
            }
        }
        else
        {
            std::cerr << "Unsuppored format\n";
            return false;
        }
    }

    return true;
}

} // visionaray
