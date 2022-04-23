// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_BRDF_H
#define VSNRAY_BRDF_H 1

#include "detail/color_conversion.h"
#include "math/constants.h"
#include "math/vector.h"
#include "fresnel.h"
#include "mdf.h"
#include "sampling.h"
#include "spectrum.h"
#include "surface_interaction.h"

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
    spectrum<U> f(vector<3, U> const& n, vector<3, U> const& wo, vector<3, U> const& wi) const
    {
        VSNRAY_UNUSED(n);
        VSNRAY_UNUSED(wi);
        VSNRAY_UNUSED(wo);

        return spectrum<U>( cd * kd * constants::inv_pi<T>() );
    }

    template <typename U, typename Interaction, typename Generator>
    VSNRAY_FUNC
    spectrum<U> sample_f(
            vector<3, T> const& n,
            vector<3, U> const& wo,
            vector<3, U>&       wi,
            U&                  pdf,
            Interaction&        inter,
            Generator&          gen
            ) const
    {
        auto w = n;
        auto v = select(
                abs(w.x) > abs(w.y),
                normalize( vector<3, U>(-w.z, U(0.0), w.x) ),
                normalize( vector<3, U>(U(0.0), w.z, -w.y) )
                );
        auto u = cross(v, w);

        auto sp = cosine_sample_hemisphere(gen.next(), gen.next());
        wi      = normalize( sp.x * u + sp.y * v + sp.z * w );

        pdf     = this->pdf(n, wo, wi);

        inter   = Interaction(surface_interaction::Diffuse);

        return f(n, wo, wi);
    }

    template <typename U>
    VSNRAY_FUNC
    U pdf(vector<3, U> const& n, vector<3, U> const& wo, vector<3, U> const& wi) const
    {
        VSNRAY_UNUSED(wo);
        return dot(n, wi) * constants::inv_pi<U>();
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
    spectrum<U> f(vector<3, U> const& n, vector<3, U> const& wo, vector<3, U> const& wi) const
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
    spectrum<U> f(vector<3, U> const& n, vector<3, U> const& wo, vector<3, U> const& wi) const
    {
        auto h = normalize(wo + wi);
        auto hdotn = max( U(0.0), dot(h, n) );

        auto spec = cs * ks;
        auto schlick = spec + (U(1.0) - spec) * pow(U(1.0) - saturate(dot(wi, h)), U(5.0));
        auto nfactor = ((exp + U(2.0)) / (U(8.0) * constants::pi<U>()));

        return spectrum<U>(schlick * nfactor * pow(hdotn, exp) );
    }

    template <typename U, typename Interaction, typename Generator>
    VSNRAY_FUNC
    spectrum<U> sample_f(
            vector<3, U> const& n,
            vector<3, U> const& wo,
            vector<3, U>&       wi,
            U&                  pdf,
            Interaction&        inter,
            Generator&          gen
            ) const
    {
        auto u1 = gen.next();
        auto u2 = gen.next();

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
        pdf = ( ((exp + U(1.0)) * pow(costheta, exp)) / (U(2.0) * constants::pi<U>() * U(4.0) * vdoth) );

        inter = Interaction(surface_interaction::GlossyReflection);

        return f(n, wo, wi);
    }

    template <typename U>
    VSNRAY_FUNC
    U pdf(vector<3, U> const& n, vector<3, U> const& wo, vector<3, U> const& wi) const
    {
        vector<3, U> h = normalize(wi + wo);
        U costheta = dot(n, h);
        U vdoth = dot(wo, h);
        return ( ((exp + U(1.0)) * pow(costheta, exp)) / (U(2.0) * constants::pi<U>() * U(4.0) * vdoth) );
    }
};


//-------------------------------------------------------------------------------------------------
// Cook Torrance reflection like in:
// Walter et al.: Microfacet Models for Refraction through Rough Surfaces
//

template <typename T, typename MDF>
class cook_torrance
{
public:

    using scalar_type   = T;

public:

    spectrum<T> ior;
    spectrum<T> absorption;
    MDF mdf;

    template <typename U>
    VSNRAY_FUNC
    spectrum<U> f(vector<3, U> const& n, vector<3, U> const& wo, vector<3, U> const& wi) const
    {
        // h is proportional to the microfacet normal
        auto h = wi + wo;

        auto ldotn = abs(dot(wi, n));
        auto vdotn = abs(dot(wo, n));

        auto active = length(h) != U(0.0) && ldotn != U(0.0) && vdotn != U(0.0);

        h = normalize(h);

        auto F = fresnel_reflectance(
                conductor_tag{},
                ior,
                absorption,
                dot(wi, h)
                );

        // "The Smith G approximates the bidirectional shadowing-masking as
        // the separable product of two monodirectional shadowing terms G1:"
        auto G = mdf.g1(n, h, wi) * mdf.g1(n, h, wo);

        auto D = mdf.d(n, h);

        auto fr = (F * G * D) / (U(4) * ldotn * vdotn);

        return select(active, fr, spectrum<U>(0.0));
    }

    template <typename U, typename Interaction, typename Generator>
    VSNRAY_FUNC
    spectrum<U> sample_f(
            vector<3, T> const& n,
            vector<3, U> const& wo,
            vector<3, U>&       wi,
            U&                  pdf,
            Interaction&        inter,
            Generator&          gen
            ) const
    {
        auto w = n;
        auto v = select(
                abs(w.x) > abs(w.y),
                normalize( vector<3, U>(-w.z, U(0.0), w.x) ),
                normalize( vector<3, U>(U(0.0), w.z, -w.y) )
                );
        auto u = cross(v, w);

        auto sp = cosine_sample_hemisphere(gen.next(), gen.next());
        wi      = normalize( sp.x * u + sp.y * v + sp.z * w );

        pdf     = this->pdf(n, wo, wi);

        inter   = Interaction(surface_interaction::GlossyReflection);

        return f(n, wo, wi);
    }

    template <typename U>
    VSNRAY_FUNC
    U pdf(vector<3, U> const& n, vector<3, U> const& wo, vector<3, U> const& wi) const
    {
        VSNRAY_UNUSED(wo);
        return dot(n, wi) * constants::inv_pi<U>();
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
    spectrum<U> f(vector<3, U> const& n, vector<3, U> const& wo, vector<3, U> const& wi) const
    {
        VSNRAY_UNUSED(n);
        VSNRAY_UNUSED(wi);
        VSNRAY_UNUSED(wo);

        return spectrum<U>(0.0);
    }

    template <typename U, typename Interaction, typename Generator>
    VSNRAY_FUNC
    spectrum<U> sample_f(
            vector<3, U> const& n,
            vector<3, U> const& wo,
            vector<3, U>&       wi,
            U&                  pdf,
            Interaction&        inter,
            Generator&          gen
            ) const
    {
        VSNRAY_UNUSED(gen);

        wi = reflect(wo, n);
        pdf = U(1.0);

        inter = Interaction(surface_interaction::SpecularReflection);

        return fresnel_reflectance(
                conductor_tag(),
                ior,
                absorption,
                abs( dot(n, wo) )
                ) * spectrum<U>(cr * kr) / abs( dot(n, wi) );
    }

    template <typename U>
    VSNRAY_FUNC
    U pdf(vector<3, U> const& n, vector<3, U> const& wo, vector<3, U> const& wi) const
    {
        VSNRAY_UNUSED(n);
        VSNRAY_UNUSED(wo);
        VSNRAY_UNUSED(wi);
        return U(0.0);
    }
};


//-------------------------------------------------------------------------------------------------
// Perfect specular transmission
//

template <typename T>
class specular_transmission
{
public:

    using scalar_type   = T;

public:

    spectrum<T> ct;
    scalar_type kt;
    spectrum<T> cr;
    scalar_type kr;
    spectrum<T> ior;

    template <typename U>
    VSNRAY_FUNC
    spectrum<U> f(vector<3, U> const& n, vector<3, U> const& wo, vector<3, U> const& wi) const
    {
        VSNRAY_UNUSED(n);
        VSNRAY_UNUSED(wi);
        VSNRAY_UNUSED(wo);

        return spectrum<U>(0.0);
    }

    template <typename U, typename Interaction, typename Generator>
    VSNRAY_FUNC
    spectrum<U> sample_f(
            vector<3, U> const& n,
            vector<3, U> const& wo,
            vector<3, U>&       wi,
            U&                  pdf,
            Interaction&        inter,
            Generator&          gen
            ) const
    {
        // IOR of material above normal direction
        spectrum<U> ior1 = spectrum<U>(1.0);
        // IOR of material below normal direction
        spectrum<U> ior2 = spectrum<U>(ior);

        U cosi = clamp(dot(n, wo), U(-1.0), U(1.0));

        auto entering = cosi > U(0.0);

        // If ray originates in 2nd medium, flip normal
        // and swap refraction indices
        vector<3, U> N = select(entering, n, -n);
        spectrum<U> etai = select(entering, ior1, ior2);
        spectrum<U> etat = select(entering, ior2, ior1);
        cosi = select(!entering, abs(cosi), cosi);

        U eta = etai[0] / etat[0]; // TODO: etaX are spectra!

        // Snell's law
        U sini = sqrt(max(U(0.0), U(1.0) - cosi * cosi));
        U sint = eta * sini;
        U cost = sqrt(max(U(0.0), U(1.0) - sint * sint));

        spectrum<U> reflectance = fresnel_reflectance(
                dielectric_tag(),
                ior1,
                ior2,
                cosi,
                cost
                );

        auto tir = sint >= U(1.0);
        reflectance = select(tir, spectrum<U>(1.0), reflectance);

        vector<3, U> refracted = refract(wo, N, eta); // NOTE: not normalized!
        vector<3, U> reflected = reflect(wo, N);

        auto u = gen.next();

        wi = select(
                u < reflectance[0],
                reflected,
                normalize(refracted)
                );

        pdf = select(
                u < reflectance[0],
                reflectance[0],
                U(1.0) - reflectance[0]
                );

        inter = select(
                u < reflectance[0],
                Interaction(surface_interaction::SpecularReflection),
                Interaction(surface_interaction::SpecularTransmission)
                );

        return select(
                u < reflectance[0],
                reflectance * spectrum<U>(cr * kr),
                (spectrum<U>(1.0) - reflectance) * spectrum<U>(ct * kt)
                ) / dot(N, wi);
    }

    template <typename U>
    VSNRAY_FUNC
    U pdf(vector<3, U> const& n, vector<3, U> const& wo, vector<3, U> const& wi) const
    {
        VSNRAY_UNUSED(n);
        VSNRAY_UNUSED(wo);
        VSNRAY_UNUSED(wi);
        return U(0.0);
    }
};

} // visionaray

#endif // VSNRAY_BRDF_H
