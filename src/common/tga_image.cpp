// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cstdint>
#include <cstring> // for memset
#include <fstream>
#include <iostream>
#include <ostream>

#include <visionaray/swizzle.h>

#include "tga_image.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// TGA header
//

#pragma pack(push, 1)

struct tga_header
{
    uint8_t  id_length;
    uint8_t  color_map_type;
    uint8_t  image_type;
    uint16_t color_map_first_entry;
    uint16_t color_map_num_entries;
    uint8_t  color_map_entry_size;
    uint16_t x_origin;
    uint16_t y_origin;
    uint16_t width;
    uint16_t height;
    uint8_t  bits_per_pixel;
    uint8_t  image_desc;
};

#pragma pack(pop)


//-------------------------------------------------------------------------------------------------
// Helper functions
//

pixel_format map_pixel_depth(uint8_t bits_per_pixel)
{
    switch (bits_per_pixel)
    {

    case 24:
        return PF_RGB8;
    case 32:
        return PF_RGBA8;
    default:
        return PF_UNSPECIFIED;

    }
}


void load_true_color_uncompressed(
        uint8_t*        dst,
        std::ifstream&  file,
        int             width,
        int             height,
        int             bytes_per_pixel,
        int             y_origin
        )
{
    int pitch = width * bytes_per_pixel;

    if (y_origin == 0)
    {
        // Origin is bottom/left corner - same as visionaray image format
        file.read(reinterpret_cast<char*>(dst), pitch * height * sizeof(uint8_t));
    }
    else if (y_origin == height)
    {
        // Origin is top/left corner - convert to bottom/left
        for (int y = 0; y < height; ++y)
        {
            auto ptr = dst + (height - 1) * pitch - y * pitch;
            file.read(reinterpret_cast<char*>(ptr), pitch * sizeof(uint8_t));
        }
    }
    else
    {
        std::cerr << "Unsupported TGA image, y-origin ("
                  << y_origin << ") is neither bottom nor top\n";
    }
}


void load_true_color_rle(
        uint8_t*        dst,
        std::ifstream&  file,
        int             width,
        int             height,
        int             bytes_per_pixel,
        int             y_origin
        )
{
    if (y_origin != 0 && y_origin != height)
    {
        std::cerr << "Unsupported TGA image, y-origin ("
                  << y_origin << ") is neither bottom nor top\n";
        return;
    }

    int pixels = 0;

    while (pixels < width * height)
    {
        uint8_t chunk_header = 0;
        file.read((char*)&chunk_header, sizeof(chunk_header));

        int current_pixel = pixels; // true if y-origin = 0
        if (y_origin == height)
        {
            int x = pixels % width;
            int y = pixels / width;
            current_pixel = (height - 1) * width - y * width + x;
        }
        auto ptr = dst + current_pixel * bytes_per_pixel;

        if (chunk_header < 128)
        {
            ++chunk_header;
            file.read(reinterpret_cast<char*>(ptr), chunk_header * bytes_per_pixel);
        }
        else
        {
            chunk_header -= 127;
            char buffer[4];
            file.read(buffer, bytes_per_pixel);
            for (uint8_t i = 0; i < chunk_header; ++i)
            {
                for (int b = 0; b < bytes_per_pixel; ++b)
                {
                    *(ptr + i * bytes_per_pixel + b) = buffer[b];
                }
            }
        }

        pixels += chunk_header;
    }
}


//-------------------------------------------------------------------------------------------------
// tga_image
//

tga_image::tga_image(std::string const& filename)
{
    std::ifstream file(filename, std::ios::in | std::ios::binary);


    // Read header

    tga_header header;
    memset(&header, 0, sizeof(header));
    file.seekg (0, file.beg);
    file.read(reinterpret_cast<char*>(&header), sizeof(header));

    width_  = static_cast<size_t>(header.width);
    height_ = static_cast<size_t>(header.height);
    format_ = map_pixel_depth(header.bits_per_pixel);
    auto pitch = header.width * (header.bits_per_pixel / 8);
    data_.resize(pitch * header.height);


    // Read image data
    file.seekg(sizeof(header) + header.id_length, file.beg);

    switch (header.image_type)
    {
    default:
        std::cerr << "Unsupported TGA image type (" << (int)header.image_type << ")\n";
        // fall-through
    case 0:
        // no image data
        width_  = 0;
        height_ = 0;
        format_ = PF_UNSPECIFIED;
        data_.resize(0);
        break;

    case 2:
        load_true_color_uncompressed(
                data_.data(),
                file,
                header.width,
                header.height,
                header.bits_per_pixel / 8,
                header.y_origin
                );
        break;

    case 10:
        load_true_color_rle(
                data_.data(),
                file,
                header.width,
                header.height,
                header.bits_per_pixel / 8,
                header.y_origin
                );
        break;

    }


    // Swizzle from BGR(A) to RGB(A)

    if (format_ == PF_RGB8)
    {
        swizzle(
                reinterpret_cast<vector<3, unorm<8>>*>(data_.data()),
                PF_RGB8,
                PF_BGR8,
                data_.size() / 3
                );
    }
    else if (format_ == PF_RGBA8)
    {
        swizzle(
                reinterpret_cast<vector<4, unorm<8>>*>(data_.data()),
                PF_RGBA8,
                PF_BGRA8,
                data_.size() / 4
                );
    }
}

} // visionaray
