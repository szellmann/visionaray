// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <ostream>
#include <string>
#include <vector>

#include <boost/algorithm/string.hpp>

#include "pnm_image.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Helper functions
//

template <typename AssignFunc>
static void load_ascii(
        uint8_t*        dst,
        std::ifstream&  file,
        size_t          pitch,
        size_t          height,
        int             max_value,
        AssignFunc      assign_func // specifies how to handle color components
        )
{
    assert(max_value < 256);    // TODO: 16-bit

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

        // Remove empty tokens and spaces
        tokens.erase(
                std::remove_if(
                        tokens.begin(),
                        tokens.end(),
                        [](std::string str) { return str.empty() || std::isspace(str[0]); }
                        ),
                tokens.end()
                );

        assert(tokens.size() == pitch);

        for (size_t x = 0; x < pitch; ++x)
        {
            int val = std::stoi(tokens[x]);
            if (max_value != 255)
            {
                double n = val / static_cast<double>(max_value); // scale down to [0..1]
                val = static_cast<int>(n * 255);                 // scale up to [0..255]
            }

            dst[y * pitch + x] = assign_func(val);
        }
    }
}

static void load_binary(
        uint8_t*        dst,
        std::ifstream&  file,
        size_t          pitch,
        size_t          height,
        int             max_value
        )
{
    assert(max_value < 256);    // TODO: 16-bit

    file.read(reinterpret_cast<char*>(dst), pitch * height);

    if (max_value != 255)
    {
        double scale = (1.0 / max_value) * 255;

        size_t size = height * pitch;
        for (size_t i = 0; i < size; ++i)
        {
            dst[i] = static_cast<uint8_t>(dst[i] * scale);
        }
    }
}

static void save_ascii(
        std::ofstream&  file,
        uint8_t const*  src,
        size_t          pitch,
        size_t          height
        )
{
    for (size_t y = 0; y < height; ++y)
    {
        for (size_t x = 0; x < pitch; ++x)
        {
            auto c = std::to_string(src[y * pitch + x]);

            // padding
            for (size_t i = 0; i < 3 - c.length(); ++i)
            {
                file << ' ';
            }

            // write color component
            file << c;

            // whitespace
            if (x + 1 < pitch)
            {
                file << ' ';
            }
            else
            {
                file << '\n';
            }
        }
    }
}

static void save_binary(
        std::ofstream&  file,
        uint8_t const*  src,
        size_t          pitch,
        size_t          height
        )
{
    file.write(reinterpret_cast<char const*>(src), pitch * height);
}


//-------------------------------------------------------------------------------------------------
// pnm_image
//

pnm_image::pnm_image(size_t width, size_t height, pixel_format format, uint8_t const* data)
    : image_base(width, height, format, data)
{
}

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

    // Header
    int header[3]; // width, height, max. value
    int index = 0;

    bool is_bitmap = (fmt == P1 || fmt == P4);

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

            if (tokens.size() > 3)
            {
                std::cerr << "Invalid pnm file header\n";
                return false;
            }

            for (auto t : tokens)
            {
                header[index++] = std::stoi(t);

                if ((is_bitmap && index > 2) || (!is_bitmap && index > 3))
                {
                    std::cerr << "Invalid pnm file header\n";
                    return false;
                }
            }

            // BitMap: width and height read ==> break
            if (is_bitmap && index == 2)
            {
                break;
            }

            // GrayMap/PixMap: width, height and max_value read ==> break
            if (!is_bitmap && index == 3)
            {
                break;
            }
        }
    }

    if (header[0] <= 0 || header[1] <= 0)
    {
        std::cerr << "Invalid image dimensions: " << header[0] << ' ' << header[1] << '\n';
        return false;
    }

    width_  = static_cast<size_t>(header[0]);
    height_ = static_cast<size_t>(header[1]);
    int max_value = header[2];

    switch (fmt)
    {
    default:
        std::cerr << "Unsupported PNM image type: P" << std::to_string(static_cast<int>(fmt)) << '\n';
        width_  = 0;
        height_ = 0;
        format_ = PF_UNSPECIFIED;
        data_.resize(0);
        return false;

    case P1:
        format_ = PF_R8;
        data_.resize(width_ * height_);

        // black or white
        load_ascii(
                data_.data(),
                file,
                width_,
                height_,
                max_value,
                [](int val)
                    {
                        assert( val == 0 || val == 1);
                        return val ? 0U : 255U;
                    }
                );
        return true;

    case P2:
        format_ = PF_R8;
        data_.resize(width_ * height_);

        // single gray component
        load_ascii(
                data_.data(),
                file,
                width_,
                height_,
                max_value,
                [](int val)
                    {
                        return static_cast<uint8_t>(val);
                    }
                );
        return true;

    case P3:
        format_ = PF_RGB8;
        data_.resize(width_ * height_ * 3);

        // RGB color components
        load_ascii(
                data_.data(),
                file,
                width_ * 3,
                height_,
                max_value,
                [](int val)
                    {
                        return static_cast<uint8_t>(val);
                    }
                );
        return true;

    case P5:
        format_ = PF_R8;
        data_.resize(width_ * height_);

        load_binary(
                data_.data(),
                file,
                width_,
                height_,
                max_value
                );
        return true;

    case P6:
        format_ = PF_RGB8;
        data_.resize(width_ * height_ * 3);

        load_binary(
                data_.data(),
                file,
                width_ * 3,
                height_,
                max_value
                );
        return true;
    }

    return false;
}

bool pnm_image::save(std::string const& filename, image_base::save_options const& options)
{
    auto it = std::find_if(
            options.begin(),
            options.end(),
            [](save_option const& opt) { return opt.first == "binary"; }
            );

    if (it == options.end())
    {
        std::cerr << "Option \"binary\" is mandatory!\n";
        return false;
    }

    bool binary = boost::any_cast<bool>(it->second);

    std::ofstream file(filename);

    // TODO:                                   P1
    if (!binary && format_ == PF_R8)        // P2
    {
        file << "P2\n";
        file << width_ << ' ' << height_ << '\n';
        file << 255 << '\n';
        save_ascii(
                file,
                data_.data(),
                width_,
                height_
                );
    }
    else if (!binary && format_ == PF_RGB8) // P3
    {
        file << "P3\n";
        file << width_ << ' ' << height_ << '\n';
        file << 255 << '\n';
        save_ascii(
                file,
                data_.data(),
                width_ * 3,
                height_
                );
    }
    // TODO:                                   P4
    else if (binary && format_ == PF_R8)    // P5
    {
        file << "P5\n";
        file << width_ << ' ' << height_ << '\n';
        file << 255 << '\n';
        save_binary(
                file,
                data_.data(),
                width_,
                height_
                );
    }
    else if (binary && format_ == PF_RGB8)  // P6
    {
        file << "P6\n";
        file << width_ << ' ' << height_ << '\n';
        file << 255 << '\n';
        save_binary(
                file,
                data_.data(),
                width_ * 3,
                height_
                );
    }
    else
    {
        std::cerr << "Unsupported image format\n";
        return false;
    }


    file.close();

    if (!file)
    {
        std::cerr << "Error writing to file\n";
        return false;
    }

    return true;
}

} // visionaray
