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

// Make a four-character constant from four characters
#define MAKE_FOURCC(C0, C1, C2, C3)     \
   (static_cast<unsigned>(C0)           \
 | (static_cast<unsigned>(C1) <<  8)    \
 | (static_cast<unsigned>(C2) << 16)    \
 | (static_cast<unsigned>(C3) << 24))
    

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

#define D3DFMT_DXT1                MAKE_FOURCC('D','X','T','1')
#define D3DFMT_DXT2                MAKE_FOURCC('D','X','T','2')
#define D3DFMT_DXT3                MAKE_FOURCC('D','X','T','3')
#define D3DFMT_DXT4                MAKE_FOURCC('D','X','T','4')
#define D3DFMT_DXT5                MAKE_FOURCC('D','X','T','5')


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


static void decode_tile_rgb_5_6_5(uint8_t* dst, uint8_t const* src)
{
    using visionaray::vec3;

    uint16_t c0 = *reinterpret_cast<uint16_t const*>(src);
    uint16_t c1 = *reinterpret_cast<uint16_t const*>(src + 2);

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

    uint32_t bitmask = *reinterpret_cast<uint32_t const*>(src + 4);
    for (int i = 0; i < 16; ++i)
    {
        int index = bitmask & 0x3;

        dst[i * 3]     = color[index].x * 255;
        dst[i * 3 + 1] = color[index].y * 255;
        dst[i * 3 + 2] = color[index].z * 255;

        bitmask >>= 2;
    }
}


static void decode_tile_alpha_dxt5(uint8_t* dst, uint8_t const* src)
{
    uint8_t a0 = src[0];
    uint8_t a1 = src[1];

    float alpha[8];

    alpha[0] = src[0] / 255.0f;
    alpha[1] = src[1] / 255.0f;

    alpha[2] = a0 > a1
        ? (6.0f * alpha[0] + 1.0f * alpha[1]) / 7.0f
        : (4.0f * alpha[0] + 1.0f * alpha[1]) / 5.0f;

    alpha[3] = a0 > a1
        ? (5.0f * alpha[0] + 2.0f * alpha[1]) / 7.0f
        : (3.0f * alpha[0] + 2.0f * alpha[1]) / 5.0f;

    alpha[4] = a0 > a1
        ? (4.0f * alpha[0] + 3.0f * alpha[1]) / 7.0f
        : (2.0f * alpha[0] + 3.0f * alpha[1]) / 5.0f;

    alpha[5] = a0 > a1
        ? (3.0f * alpha[0] + 4.0f * alpha[1]) / 7.0f
        : (1.0f * alpha[0] + 4.0f * alpha[1]) / 5.0f;

    alpha[6] = a0 > a1
        ? (2.0f * alpha[0] + 5.0f * alpha[1]) / 7.0f
        : 0.0f;

    alpha[7] = a0 > a1
        ? (1.0f * alpha[0] + 6.0f * alpha[1]) / 7.0f
        : 1.0f;

    uint64_t bitmask = *reinterpret_cast<uint64_t const*>(src + 2);
    for (int i = 0; i < 16; ++i)
    {
        int index = bitmask & 0x7;

        dst[i] = alpha[index] * 255;

        bitmask >>= 3;
    }
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

    width_ = static_cast<int>(header.width);
    height_ = static_cast<int>(header.height);

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
            decode_tile_rgb_5_6_5(data_.data() + i * 2 * 3, bytes.data() + i);
        }

        return true;
    }

    if (format == DDS_PF_DXT5)
    {
        std::vector<uint8_t> bytes(header.pitch_or_linear_size);
        file.read(reinterpret_cast<char*>(bytes.data()), bytes.size());

        format_ = PF_RGBA8;

        auto pitch = header.width * 4;
        data_.resize(pitch * header.height);

        for (size_t i = 0; i < bytes.size(); i += 16)
        {
            uint8_t rgb[48];
            uint8_t alpha[16];
            decode_tile_alpha_dxt5(alpha, bytes.data() + i);
            decode_tile_rgb_5_6_5(rgb, bytes.data() + i + 8);

            for (int j = 0; j < 16; ++j)
            {
                size_t index = (i + j) * 4;
                data_[index]     = rgb[j * 3];
                data_[index + 1] = rgb[j * 3 + 1];
                data_[index + 2] = rgb[j * 3 + 2];
                data_[index + 3] = alpha[j];
            }
        }

        return true;
    }

    return false;
}

} // visionaray
