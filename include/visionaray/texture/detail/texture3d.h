// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_TEXTURE_DETAIL_TEXTURE3D_H
#define VSNRAY_TEXTURE_DETAIL_TEXTURE3D_H 1

#include <cstddef>

#include "texture_common.h"


namespace visionaray
{

template <typename Base, typename T>
class texture_iface<Base, T, 3> : public Base
{
public:

    using ref_type = texture_ref<T, 3>;

    using base_type = Base;
    using value_type = T;

public:

    texture_iface() = default;

    texture_iface(size_t w, size_t h, size_t d)
        : Base(w * h * d)
        , width_(w)
        , height_(h)
        , depth_(d)
    {
    }

    template <typename B2>
    texture_iface(texture_iface<B2, T, 3> const& rhs)
        : Base(rhs)
        , width_(rhs.width())
        , height_(rhs.height())
        , depth_(rhs.depth())
    {
    }


    vector<3, size_t> size() const
    {
        return vector<3, size_t>(width_, height_, depth_);
    }

    size_t width() const { return width_; }
    size_t height() const { return height_; }
    size_t depth() const { return depth_; }

    operator bool() const
    {
        return static_cast<bool>(static_cast<Base>(*this)) && width_ > 0 && height_ > 0 && depth_ > 0;
    }

private:

    size_t width_;
    size_t height_;
    size_t depth_;

};

} // visionaray

#endif // VSNRAY_TEXTURE_DETAIL_TEXTURE3D_H
