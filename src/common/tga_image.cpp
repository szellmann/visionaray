// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cassert>
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
        bool            flip_y
        )
{
    int pitch = width * bytes_per_pixel;

    if (!flip_y)
    {
        // Origin is bottom/left corner - same as visionaray image format
        file.read(reinterpret_cast<char*>(dst), pitch * height * sizeof(uint8_t));
    }
    else
    {
        // Origin is top/left corner - convert to bottom/left
        for (int y = 0; y < height; ++y)
        {
            auto ptr = dst + (height - 1) * pitch - y * pitch;
            file.read(reinterpret_cast<char*>(ptr), pitch * sizeof(uint8_t));
        }
    }
}


void load_true_color_rle(
        uint8_t*        dst,
        std::ifstream&  file,
        int             width,
        int             height,
        int             bytes_per_pixel,
        bool            flip_y
        )
{
    assert(width > 0 && height > 0);

    int pixels  = 0;
    int x       = 0;
    int y       = flip_y ? height - 1 : 0;
    int yinc    = flip_y ? -1 : 1;

    while (pixels < width * height)
    {
        uint8_t hdr = 0;

        file.read((char*)&hdr, sizeof(hdr));

        // Run-length packet or raw packet?
        bool rle = (hdr & 0x80) != 0;

        // Get number of pixels or repetition count
        int count = 1 + (hdr & 0x7f);

        pixels += count;

        char buffer[4];
        if (rle)
        {
            // Read the RGB value for the next COUNT pixels
            file.read(buffer, bytes_per_pixel);
        }

        while (count-- > 0)
        {
            auto p = dst + (x + y * width) * bytes_per_pixel;

            if (rle)
                memcpy(p, buffer, bytes_per_pixel);
            else
                file.read((char*)p, bytes_per_pixel);

            // Adjust current pixel position
            ++x;
            if (x >= width)
            {
                x = 0;
                y += yinc;
            }
        }
    }
}


//-------------------------------------------------------------------------------------------------
// tga_image
//

bool tga_image::load(std::string const& filename)
{
    std::ifstream file(filename, std::ios::in | std::ios::binary);


    // Read header

    tga_header header;

    memset(&header, 0, sizeof(header));

    file.seekg (0, file.beg);
    //
    // XXX:
    // Values are always stored little-endian...
    //
    file.read(reinterpret_cast<char*>(&header), sizeof(header));


    // Check header

    if (header.width <= 0 || header.height <= 0)
    {
        std::cerr << "Invalid image dimensions (" << header.width << " x " << header.height << ")\n";
        return false;
    }


    // Allocate storage

    width_  = static_cast<size_t>(header.width);
    height_ = static_cast<size_t>(header.height);
    format_ = map_pixel_depth(header.bits_per_pixel);
    auto pitch = header.width * (header.bits_per_pixel / 8);
    data_.resize(pitch * header.height);


    // Read image data

    file.seekg(sizeof(header) + header.id_length, file.beg);

    // Bit 5 specifies the screen origin:
    // 0 = Origin in lower left-hand corner.
    // 1 = Origin in upper left-hand corner.
    bool flip_y = (header.image_desc & (1 << 5)) != 0;

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
                flip_y
                );
        break;

    case 10:
        load_true_color_rle(
                data_.data(),
                file,
                header.width,
                header.height,
                header.bits_per_pixel / 8,
                flip_y
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
    else
    {
        return false;
    }

    return true;
}

} // visionaray
