// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_GENERIC_PRIM_H
#define VSNRAY_GENERIC_PRIM_H

#include <stdexcept>

#include "math/primitive.h"

namespace visionaray
{

struct generic_prim
{
    unsigned type_;

    generic_prim()
    {
    }

    /* implicit */ generic_prim(basic_triangle<3, float> const& tri)
        : type_(detail::TrianglePrimitive)
        , triangle(tri)
    {
    }

    /* implicit */ generic_prim(basic_sphere<float> const& sph)
        : type_(detail::SpherePrimitive)
        , sphere(sph)
    {
    }

    void operator=(basic_triangle<3, float> const& tri)
    {
        type_ = detail::TrianglePrimitive;
        triangle = tri;
    }

    void operator=(basic_sphere<float> const& sph)
    {
        type_ = detail::SpherePrimitive;
        sphere = sph;
    }

    union
    {
        basic_triangle<3, float> triangle;
        basic_sphere<float> sphere;
    };
};


template <typename T>
inline hit_record<basic_ray<T>, primitive<unsigned>> intersect(basic_ray<T> const& ray, generic_prim const& p)
{
    if (p.type_ == detail::TrianglePrimitive)
    {
        return intersect(ray, p.triangle);
    }
    else if (p.type_ == detail::SpherePrimitive)
    {
        return intersect(ray, p.sphere);
    }
    else
    {
        throw std::runtime_error("primitive type unspecified");
    }
}

} // visionaray


#endif // VSNRAY_GENERIC_PRIM_H


