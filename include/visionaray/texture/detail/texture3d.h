// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_TEXTURE_TEXTURE3D_H
#define VSNRAY_TEXTURE_TEXTURE3D_H

#include <cstddef>

#include "texture_common.h"


namespace visionaray
{


template <typename Base, typename T, tex_read_mode ReadMode>
class texture_iface<Base, T, ReadMode, 3> : public Base
{
public:

    using base_type = Base;
    using value_type = T;

    using Base::Base;

    static const size_t dimensions = 3;

public:

    texture_iface() = default;

    texture_iface(size_t w, size_t h, size_t d)
        : width_(w)
        , height_(h)
        , depth_(d)
    {
    }

    template <typename B2>
    texture_iface(texture_iface<B2, T, ReadMode, 3> const& rhs)
        : Base(rhs)
        , width_(rhs.width())
        , height_(rhs.height())
        , depth_(rhs.depth())
    {
    }


    value_type& operator()(size_t x, size_t y, size_t z)
    {
        return base_type::data()[z * width_ * height_ + y * width_ + x];
    }

    value_type const& operator()(size_t x, size_t y, size_t z) const
    {
        return base_type::data()[z * width_ * height_ + y * width_ + x];
    }


    size_t size() const { return width_ * height_ * depth_; }

    size_t width() const { return width_; }
    size_t height() const { return height_; }
    size_t depth() const { return depth_; }

    void resize(size_t w, size_t h, size_t d)
    {
        width_ = w;
        height_ = h;
        depth_ = d;
    }

private:

    size_t width_;
    size_t height_;
    size_t depth_;

};


} // visionaray


#endif // VSNRAY_TEXTURE_TEXTURE3D_H
