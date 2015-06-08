// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_TEXTURE_TEXTURE1D_H
#define VSNRAY_TEXTURE_TEXTURE1D_H

#include <cstddef>

#include "texture_common.h"


namespace visionaray
{


template <typename Base, typename T, tex_read_mode ReadMode>
class texture_iface<Base, T, ReadMode, 1> : public Base
{
public:

    using base_type = Base;
    using value_type = T;

    using Base::Base;

    static const size_t dimensions = 1;

public:

    texture_iface() = default;

    texture_iface(size_t w)
        : width_(w)
    {
    }

    template <typename B2>
    texture_iface(texture_iface<B2, T, ReadMode, 1> const& rhs)
        : Base(rhs)
        , width_(rhs.width())
    {
    }


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

private:

    size_t width_;

};


} // visionaray


#endif // VSNRAY_TEXTURE_TEXTURE1D_H
