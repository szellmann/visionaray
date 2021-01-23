// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_MAKE_TEXTURE_H
#define VSNRAY_COMMON_MAKE_TEXTURE_H 1

#include <iostream>
#include <memory>
#include <ostream>
#include <type_traits>

#include <visionaray/math/unorm.h>
#include <visionaray/math/vector.h>
#include <visionaray/texture/texture.h>
#include <visionaray/pixel_format.h>

#include "image.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Overload with 4x unorm8!
//

template <
    typename Texture,
    typename = typename std::enable_if<std::is_same<typename Texture::value_type, vector<4, unorm<8>>>::value>::type
    >
inline void make_texture(Texture& tex, image const& img)
{
    tex.set_address_mode(Wrap);
    tex.set_filter_mode(Linear);
    tex.set_color_space(sRGB);

    if (img.format() == PF_RGB32F)
    {
        // Down-convert to 8-bit, add alpha=1.0
        auto data_ptr = reinterpret_cast<vector<3, float> const*>(img.data());
        tex.reset(data_ptr, PF_RGB32F, PF_RGBA8, AlphaIsOne);
    }
    else if (img.format() == PF_RGBA32F)
    {
        // Down-convert to 8-bit
        auto data_ptr = reinterpret_cast<vector<4, float> const*>(img.data());
        tex.reset(data_ptr, PF_RGBA32F, PF_RGBA8);
    }
    else if (img.format() == PF_RGB16UI)
    {
        // Down-convert to 8-bit, add alpha=1.0
        auto data_ptr = reinterpret_cast<vector<3, unorm<16>> const*>(img.data());
        tex.reset(data_ptr, PF_RGB16UI, PF_RGBA8, AlphaIsOne);
    }
    else if (img.format() == PF_RGBA16UI)
    {
        // Down-convert to 8-bit
        auto data_ptr = reinterpret_cast<vector<4, unorm<16>> const*>(img.data());
        tex.reset(data_ptr, PF_RGBA16UI, PF_RGBA8);
    }
    else if (img.format() == PF_R8)
    {
        // Let RGB=R and add alpha=1.0
        auto data_ptr = reinterpret_cast<unorm< 8> const*>(img.data());
        tex.reset(data_ptr, PF_R8, PF_RGBA8, AlphaIsOne);
    }
    else if (img.format() == PF_RGB8)
    {
        // Add alpha=1.0
        auto data_ptr = reinterpret_cast<vector<3, unorm< 8>> const*>(img.data());
        tex.reset(data_ptr, PF_RGB8, PF_RGBA8, AlphaIsOne);
    }
    else if (img.format() == PF_RGBA8)
    {
        // "Native" texture format
        auto data_ptr = reinterpret_cast<vector<4, unorm< 8>> const*>(img.data());
        tex.reset(data_ptr);
    }
    else
    {
        std::cerr << "Warning: unsupported pixel format\n";
    }
}


//-------------------------------------------------------------------------------------------------
// Overload with 4x float!
//
template <
    typename Texture,
    typename = typename std::enable_if<std::is_same<typename Texture::value_type, vector<4, float>>::value>::type,
    typename = void
    >
inline void make_texture(Texture& tex, image const& img)
{
    if (img.format() == PF_RGB32F)
    {
        // Add alpha=1.0
        auto data_ptr = reinterpret_cast<vector<3, float> const*>(img.data());
        tex.reset(data_ptr, PF_RGB32F, PF_RGBA32F, AlphaIsOne);
    }
    else if (img.format() == PF_RGBA32F)
    {
        // "Native" texture format
        auto data_ptr = reinterpret_cast<vector<4, float> const*>(img.data());
        tex.reset(data_ptr);
    }
    else if (img.format() == PF_RGB8)
    {
        // Up-convert to float and add alpha=1.0
        auto data_ptr = reinterpret_cast<vector<3, unorm< 8>> const*>(img.data());
        tex.reset(data_ptr, PF_RGB8, PF_RGBA32F, AlphaIsOne);
    }
    else if (img.format() == PF_RGBA8)
    {
        // Up-convert to float
        auto data_ptr = reinterpret_cast<vector<4, unorm< 8>> const*>(img.data());
        tex.reset(data_ptr, PF_RGBA8, PF_RGBA32F);
    }
    else
    {
        std::cerr << "Warning: unsupported pixel format\n";
    }
}

} // visionaray

#endif // VSNRAY_COMMON_MAKE_TEXTURE_H
