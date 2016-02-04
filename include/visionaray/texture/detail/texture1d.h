// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_TEXTURE_TEXTURE1D_H
#define VSNRAY_TEXTURE_TEXTURE1D_H 1

#include <cstddef>
#include <type_traits>

#include "texture_common.h"


namespace visionaray
{

template <typename Base, typename T, tex_read_mode ReadMode>
class texture_iface<Base, T, ReadMode, 1> : public Base
{
public:

    using base_type = Base;
    using value_type = T;

#ifdef VSNRAY_CXX_HAS_INHERITING_CONSTRUCTORS
    using Base::Base;
#endif

public:

    texture_iface() = default;

    texture_iface(size_t w)
        : Base(w)
        , width_(w)
    {
    }

    template <typename B2>
    texture_iface(texture_iface<B2, T, ReadMode, 1> const& rhs)
        : Base(rhs)
        , width_(rhs.width())
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

    value_type& operator()(size_t x)
    {
        return base_type::data()[x];
    }

    value_type const& operator()(size_t x) const
    {
        return base_type::data()[x];
    }


    size_t size() const { return width_; }

    size_t width() const { return width_; }

    void resize(size_t w)
    {
        width_ = w;
    }

private:

    size_t width_;

};

} // visionaray

#endif // VSNRAY_TEXTURE_TEXTURE1D_H
