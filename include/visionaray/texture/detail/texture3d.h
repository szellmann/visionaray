// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_TEXTURE_TEXTURE3D_H
#define VSNRAY_TEXTURE_TEXTURE3D_H 1

#include <cstddef>
#include <type_traits>

#include "texture_common.h"


namespace visionaray
{

template <typename Base, typename T, tex_read_mode ReadMode>
class texture_iface<Base, T, ReadMode, 3> : public Base
{
public:

    using base_type = Base;
    using value_type = T;

#ifdef VSNRAY_CXX_HAS_INHERITING_CONSTRUCTORS
    using Base::Base;
#endif

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
    texture_iface(texture_iface<B2, T, ReadMode, 3> const& rhs)
        : Base(rhs)
        , width_(rhs.width())
        , height_(rhs.height())
        , depth_(rhs.depth())
    {
    }

#ifndef VSNRAY_CXX_HAS_INHERITING_CONSTRUCTORS
    template <
        typename A,
        typename ...Args,
        typename = typename std::enable_if<std::is_constructible<Base, A, Args...>::value>::type
        >
    texture_iface(A&& a, Args&&... args)
        : Base(std::forward<A>(a), std::forward<Args>(args)...)
    {
    }
#endif

    value_type& operator()(size_t x, size_t y, size_t z)
    {
        return base_type::data()[z * width_ * height_ + y * width_ + x];
    }

    value_type const& operator()(size_t x, size_t y, size_t z) const
    {
        return base_type::data()[z * width_ * height_ + y * width_ + x];
    }


    vector<3, size_t> size() const
    {
        return vector<3, size_t>(width_, height_, depth_);
    }

    size_t width() const { return width_; }
    size_t height() const { return height_; }
    size_t depth() const { return depth_; }

    void resize(size_t w, size_t h, size_t d)
    {
        width_ = w;
        height_ = h;
        depth_ = d;
        Base::reset(w * h * d);
    }

private:

    size_t width_  = 0U;
    size_t height_ = 0U;
    size_t depth_  = 0U;

};

} // visionaray

#endif // VSNRAY_TEXTURE_TEXTURE3D_H
