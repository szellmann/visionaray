// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_TEXTURE_DETAIL_TEXTURE2D_H
#define VSNRAY_TEXTURE_DETAIL_TEXTURE2D_H 1

#include <cstddef>

#include "texture_common.h"


namespace visionaray
{

template <typename Base, typename T>
class texture_iface<Base, T, 2> : public Base
{
public:

    using ref_type = texture_ref<T, 2>;

    using base_type = Base;
    using value_type = T;

public:

    texture_iface() = default;

    texture_iface(size_t w, size_t h)
        : Base(w * h)
        , width_(w)
        , height_(h)
    {
    }

    template <typename B2>
    texture_iface(texture_iface<B2, T, 2> const& rhs)
        : Base(rhs)
        , width_(rhs.width())
        , height_(rhs.height())
    {
    }

    value_type& operator()(size_t x, size_t y)
    {
        return base_type::data()[y * width_ + x];
    }

    value_type const& operator()(size_t x, size_t y) const
    {
        return base_type::data()[y * width_ + x];
    }


    vector<2, size_t> size() const
    {
        return vector<2, size_t>(width_, height_);
    }

    size_t width() const { return width_; }
    size_t height() const { return height_; }

private:

    size_t width_;
    size_t height_;

};

} // visionaray

#endif // VSNRAY_TEXTURE_DETAIL_TEXTURE2D_H
