// This file is distributed under the MIT license.
// See the LICENSE file for details.

//-------------------------------------------------------------------------------------------------
// This implementation is based on the DDS loader written by Jon Watte.
// Original license follows.

/* DDS loader written by Jon Watte 2002 */
/* Permission granted to use freely, as long as Jon Watte */
/* is held harmless for all possible damages resulting from */
/* your use or failure to use this code. */
/* No warranty is expressed or implied. Use at your own risk, */
/* or not at all. */

#include <fstream>
#include <iostream>
#include <ostream>
#include <vector>

#include <visionaray/math/io.h>
#include <visionaray/math/vector.h>

#include "dds_image.h"

#define DDS_MAGIC 0x20534444

// dds_header.flags
#define DDSD_CAPS                  0x00000001 
#define DDSD_HEIGHT                0x00000002 
#define DDSD_WIDTH                 0x00000004 
#define DDSD_PITCH                 0x00000008 
#define DDSD_PIXELFORMAT           0x00001000 
#define DDSD_MIPMAPCOUNT           0x00020000 
#define DDSD_LINEARSIZE            0x00080000 
#define DDSD_DEPTH                 0x00800000 

#define DDPF_ALPHAPIXELS           0x00000001 
#define DDPF_FOURCC                0x00000004 
#define DDPF_INDEXED               0x00000020 
#define DDPF_RGB                   0x00000040 

// dds_header.caps.caps1
#define DDSCAPS_COMPLEX            0x00000008 
#define DDSCAPS_TEXTURE            0x00001000 
#define DDSCAPS_MIPMAP             0x00400000 

// dds_header.caps.caps2
#define DDSCAPS2_CUBEMAP           0x00000200 
#define DDSCAPS2_CUBEMAP_POSITIVEX 0x00000400 
#define DDSCAPS2_CUBEMAP_NEGATIVEX 0x00000800 
#define DDSCAPS2_CUBEMAP_POSITIVEY 0x00001000 
#define DDSCAPS2_CUBEMAP_NEGATIVEY 0x00002000 
#define DDSCAPS2_CUBEMAP_POSITIVEZ 0x00004000 
#define DDSCAPS2_CUBEMAP_NEGATIVEZ 0x00008000 
#define DDSCAPS2_VOLUME            0x00200000 

#define D3DFMT_DXT1                '1TXD'
#define D3DFMT_DXT2                '2TXD'
#define D3DFMT_DXT3                '3TXD'
#define D3DFMT_DXT4                '4TXD'
#define D3DFMT_DXT5                '5TXD'


struct dds_header
{
    unsigned magic;
    unsigned size;
    unsigned flags;
    unsigned height;
    unsigned width;
    unsigned pitch_or_linear_size;
    unsigned depth;
    unsigned mip_map_count;
    unsigned reserved1[11];

    struct pixel_format_t
    {
        unsigned size;
        unsigned flags;
        unsigned four_cc;
        unsigned rgb_bit_count;
        unsigned r_bit_mask;
        unsigned g_bit_mask;
        unsigned b_bit_mask;
        unsigned alpha_bit_mask;
    };

    pixel_format_t pixel_format;

    struct caps_t
    {
        unsigned caps1;
        unsigned caps2;
        unsigned ddsx;
        unsigned reserved;
    };

    caps_t caps;

    unsigned reserved2;
};

struct dds_load_info
{
    bool compressed;
    bool swap;
    bool palette;
    unsigned div_size;
    unsigned block_bytes;
};

enum dds_pixel_format
{
    DDS_PF_DXT1,
    DDS_PF_DXT3,
    DDS_PF_DXT5,
    DDS_PF_BGRA8,
    DDS_PF_BGR8,
    DDS_PF_BGR5A1,
    DDS_PF_BGR565,
    DDS_PF_INDEX8,
    DDS_PF_UNKNOWN,
};

static dds_pixel_format get_pixel_format(dds_header::pixel_format_t pf)
{
    if ((pf.flags & DDPF_FOURCC) && (pf.four_cc == D3DFMT_DXT1))
    {
        return DDS_PF_DXT1;
    }

    if ((pf.flags & DDPF_FOURCC) && (pf.four_cc == D3DFMT_DXT3))
    {
        return DDS_PF_DXT3;
    }

    if ((pf.flags & DDPF_FOURCC) && (pf.four_cc == D3DFMT_DXT5))
    {
        return DDS_PF_DXT5;
    }

    if ((pf.flags & DDPF_RGB) && (pf.flags & DDPF_ALPHAPIXELS) &&
        (pf.rgb_bit_count == 32) && (pf.r_bit_mask == 0xFF0000) &&
        (pf.g_bit_mask == 0xFF00) && (pf.b_bit_mask == 0xFF) &&
        (pf.alpha_bit_mask == 0xFF000000u))
    {
        return DDS_PF_BGRA8;
    }

    if ((pf.flags & DDPF_RGB) && !(pf.flags & DDPF_ALPHAPIXELS) &&
        (pf.rgb_bit_count == 24) && (pf.r_bit_mask == 0xFF0000) &&
        (pf.g_bit_mask == 0xFF00) && (pf.b_bit_mask == 0xFF))
    {
        return DDS_PF_BGR5A1;
    }

    if ((pf.flags & DDPF_RGB) && (pf.flags & DDPF_ALPHAPIXELS) &&
        (pf.rgb_bit_count == 16) && (pf.r_bit_mask == 0x00007C00) &&
        (pf.g_bit_mask == 0x000003E0) && (pf.b_bit_mask == 0x0000001F) &&
        (pf.alpha_bit_mask == 0x00008000))
    {
        return DDS_PF_BGR565;
    }

    if ((pf.flags & DDPF_INDEXED) && (pf.rgb_bit_count == 8))
    {
        return DDS_PF_INDEX8;
    }

    return DDS_PF_UNKNOWN;
}


namespace visionaray
{

bool dds_image::load(std::string const& filename)
{
    std::ifstream file(filename, std::ios::in | std::ios::binary);


    // Read header

    dds_header header;

    memset(&header, 0, sizeof(header));

    file.seekg (0, file.beg);
    file.read(reinterpret_cast<char*>(&header), sizeof(header));

    if (header.magic != DDS_MAGIC || header.size != 124 ||
        !(header.flags & DDSD_PIXELFORMAT) || !(header.flags & DDSD_CAPS))
    {
        std::cerr << "DDS: file error\n";
        return false;
    }

    width_ = header.width;
    height_ = header.height;

    auto format = get_pixel_format(header.pixel_format);
    if (format == DDS_PF_UNKNOWN)
    {
        std::cerr << "DDS: unknown pixel format\n";
        return false;
    }

    if (format == DDS_PF_DXT1)
    {
        std::vector<uint8_t> bytes(header.pitch_or_linear_size);
        file.read(reinterpret_cast<char*>(bytes.data()), bytes.size());

        format_ = PF_RGB8;

        auto pitch = header.width * 3;
        data_.resize(pitch * header.height);

        for (size_t i = 0; i < bytes.size(); i += 8)
        {
            uint16_t c0 = *reinterpret_cast<uint16_t*>(&bytes[i]);
            uint16_t c1 = *reinterpret_cast<uint16_t*>(&bytes[i + 2]);

            vec3 color[4];

            color[0] = vec3(
                ((c0 & 0xF800) >> 11) / 32.0f,
                ((c0 & 0x7E0) >> 5)   / 64.0f,
                (c0 & 0x1F)           / 32.0f
                );

            color[1] = vec3(
                ((c1 & 0xF800) >> 11) / 32.0f,
                ((c1 & 0x7E0) >> 5)   / 64.0f,
                (c1 & 0x1F)           / 32.0f
                );

            color[2] = c0 > c1
                ? (2.0f * color[0] + color[1]) / 3.0f
                : (color[0] + color[1]) / 2.0f;

            color[3] = c0 > c1
                ? (color[0] + 2.0f * color[1]) / 3.0f
                : vec3(0.0f);

            uint32_t bitmask = *reinterpret_cast<uint32_t*>(&bytes[i + 4]);
            for (int j = i * 2; j < i * 2 + 16; ++j)
            {
                int index = bitmask & 0x3;

                data_[j * 3]     = color[index].x * 255;
                data_[j * 3 + 1] = color[index].y * 255;
                data_[j * 3 + 2] = color[index].z * 255;

                bitmask >> 2;
            }
        }

        return true;
    }

    return false;
}

} // visionaray
