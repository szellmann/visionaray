// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_BRDF_H
#define VSNRAY_BRDF_H

#include "math/math.h"
#include "mc.h"

namespace visionaray
{

template <typename T>
class lambertian
{
public:

    typedef T scalar_type;
    typedef vector<3, T> color_type;

    scalar_type kd;
    color_type cd;

    template <typename U>
    VSNRAY_FUNC
    vector<3, U> f(vector<3, T> const& n, vector<3, U> const& wo, vector<3, U> const& wi) const
    {
        VSNRAY_UNUSED(n);
        VSNRAY_UNUSED(wi);
        VSNRAY_UNUSED(wo);

        return vector<3, U>( cd * kd * constants::inv_pi<T>() );
    }

    template <typename U, typename S /* sampler */>
    VSNRAY_FUNC
    vector<3, U> sample_f(vector<3, T> const& n, vector<3, U> const& wo, vector<3, U>& wi, U& pdf, S& sampler)
    {
        auto w  = n;
        auto v  = normalize( cross(vector<3, U>(0.0001, 1.0, 0.0001), w) );
        auto u  = cross(v, w);

        auto sp = sample_hemisphere(sampler);
        wi      = normalize( sp.x * u + sp.y * v + sp.z * w );

        pdf     = dot(n, wi) * constants::inv_pi<U>();

        return f(n, wi, wo);
    }

};

template <typename T>
class specular
{
public:

    typedef T scalar_type;
    typedef vector<3, T> color_type;

    color_type  cs;
    scalar_type ks;
    scalar_type exp;

    template <typename U>
    VSNRAY_FUNC
    vector<3, U> f(vector<3, T> const& n, vector<3, U> const& wo, vector<3, U> const& wi) const
    {
        auto r = reflect(-wi, n);
        auto rdotv = dot(r, wo);
        auto mask = rdotv > U(0.0);

        auto I = cs * ks * ((exp + U(2.0)) / constants::two_pi<U>()) * pow(rdotv, exp);

        return select(mask, I, vector<3, U>(0.0));
    }

};

}

#endif // VSNRAY_BRDF_H
