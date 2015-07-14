// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_BRDF_H
#define VSNRAY_BRDF_H 1

#include "math/math.h"
#include "fresnel.h"
#include "mc.h"
#include "spectrum.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Lambertian reflection
//

template <typename T>
class lambertian
{
public:

    using scalar_type   = T;

public:

    scalar_type kd;
    spectrum<T> cd;

    template <typename U>
    VSNRAY_FUNC
    spectrum<U> f(vector<3, T> const& n, vector<3, U> const& wo, vector<3, U> const& wi) const
    {
        VSNRAY_UNUSED(n);
        VSNRAY_UNUSED(wi);
        VSNRAY_UNUSED(wo);

        return spectrum<U>( cd * kd * constants::inv_pi<T>() );
    }

    template <typename U, typename S /* sampler */>
    VSNRAY_FUNC
    spectrum<U> sample_f(vector<3, T> const& n, vector<3, U> const& wo, vector<3, U>& wi, U& pdf, S& sampler) const
    {
        auto w = n;
        auto v = select(
                abs(w.x) > abs(w.y),
                normalize( vector<3, U>(-w.z, U(0.0), w.x) ),
                normalize( vector<3, U>(U(0.0), w.z, -w.y) )
                );
        auto u = cross(v, w);

        auto sp = cosine_sample_hemisphere(sampler.next(), sampler.next());
        wi      = normalize( sp.x * u + sp.y * v + sp.z * w );

        pdf     = dot(n, wi) * constants::inv_pi<U>() * heavyside( dot(n, wo) );

        return f(n, wo, wi);
    }

};


//-------------------------------------------------------------------------------------------------
// Phong reflection
//

template <typename T>
class phong
{
public:

    using scalar_type   = T;

public:

    spectrum<T> cs;
    scalar_type ks;
    scalar_type exp;

    template <typename U>
    VSNRAY_FUNC
    spectrum<U> f(vector<3, T> const& n, vector<3, U> const& wo, vector<3, U> const& wi) const
    {
        auto r = reflect(wo, n);
        auto rdotl = max( U(0.0), dot(r, wi) );

        return  spectrum<U>( cs * ks * ((exp + U(2.0)) / constants::two_pi<U>()) * pow(rdotl, exp) );
    }

};


//-------------------------------------------------------------------------------------------------
// Blinn reflection
//

template <typename T>
class blinn
{
public:

    using scalar_type   = T;

public:

    spectrum<T> cs;
    scalar_type ks;
    scalar_type exp;

    template <typename U>
    VSNRAY_FUNC
    spectrum<U> f(vector<3, T> const& n, vector<3, U> const& wo, vector<3, U> const& wi) const
    {
        auto h = normalize(wo + wi);
        auto hdotn = max( U(0.0), dot(h, n) );

        return spectrum<U>( cs * ks * ((exp + U(2.0)) / (U(8.0) * constants::pi<U>())) * pow(hdotn, exp) );
    }

    template <typename U, typename S /* sampler */>
    VSNRAY_FUNC
    spectrum<U> sample_f(vector<3, T> const& n, vector<3, U> const& wo, vector<3, U>& wi, U& pdf, S& sampler) const
    {
        auto u1 = sampler.next();
        auto u2 = sampler.next();

        auto costheta = pow(u1, U(1.0) / (exp + U(1.0)));
        auto sintheta = sqrt( max(U(0.0), U(1.0) - costheta * costheta) );
        auto phi = u2 * constants::two_pi<U>();

        auto w = n;
        auto v = select(
                abs(w.x) > abs(w.y),
                normalize( vector<3, U>(-w.z, U(0.0), w.x) ),
                normalize( vector<3, U>(U(0.0), w.z, -w.y) )
                );
        auto u = cross(v, w);

        vector<3, U> h = normalize( sintheta * cos(phi) * u + sintheta * sin(phi) * v + costheta * w );

        wi = reflect(wo, h);

        auto vdoth = dot(wo, h);
        pdf = ( ((exp + U(1.0)) * pow(costheta, exp)) / (U(2.0) * constants::pi<U>() * U(4.0) * vdoth) ) * heavyside( dot(n, wo) );

        return f(n, wo, wi);
    }
};


//-------------------------------------------------------------------------------------------------
// Perfect specular reflection
//

template <typename T>
class specular_reflection
{
public:

    using scalar_type   = T;

public:

    spectrum<T> cr;
    scalar_type kr;
    spectrum<T> ior;
    spectrum<T> absorption;

    template <typename U>
    VSNRAY_FUNC
    spectrum<U> f(vector<3, T> const& n, vector<3, U> const& wo, vector<3, U> const& wi) const
    {
        VSNRAY_UNUSED(n);
        VSNRAY_UNUSED(wi);
        VSNRAY_UNUSED(wo);

        return spectrum<U>(0.0);
    }

    template <typename U, typename Sampler>
    VSNRAY_FUNC
    spectrum<U> sample_f(
            vector<3, T> const& n,
            vector<3, U> const& wo,
            vector<3, U>& wi,
            U& pdf,
            Sampler& sampler
            ) const
    {
        VSNRAY_UNUSED(sampler);

        wi = reflect(wo, n);
        pdf = U(1.0);

        return fresnel_reflectance(
                conductor_tag(),
                ior,
                absorption,
                abs( dot(n, wo) )
                ) * spectrum<U>(cr * kr) / abs( dot(n, wi) );
    }

};

} // visionaray

#endif // VSNRAY_BRDF_H
