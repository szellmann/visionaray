// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_MATERIAL_H
#define VSNRAY_MATERIAL_H

#include "detail/macros.h"
#include "math/math.h"
#include "brdf.h"
#include "shade_record.h"


namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Emissive material, no BRDFs
//

template <typename T>
class emissive
{
public:

    typedef T scalar_type;
    typedef vector<3, T> color_type;

    template <typename SR, typename U, typename S /* sampler */>
    VSNRAY_FUNC
    vector<3, U> sample(SR const& shade_rec, vector<3, U>& refl_dir, U& pdf, S& sampler)
    {
        VSNRAY_UNUSED(refl_dir); // TODO?
        VSNRAY_UNUSED(sampler);
        pdf = U(1.0);
        return shade(shade_rec);
    }

    template <typename L, typename U>
    VSNRAY_FUNC
    vector<3, U> shade(shade_record<L, U> const& sr) const
    {
        return select( dot(sr.normal, sr.view_dir) >= U(0.0), ce_ * ls_, vector<3, U>(0.0, 0.0, 0.0) );
    }

    template <typename L, typename C, typename U>
    VSNRAY_FUNC
    vector<3, U> shade(shade_record<L, C, U> const& sr) const
    {
        return select( dot(sr.normal, sr.view_dir) >= U(0.0), ce_ * ls_, vector<3, U>(0.0, 0.0, 0.0) );
    }


    VSNRAY_FUNC void set_ce(color_type const& ce)
    {
        ce_ = ce;
    }

    VSNRAY_FUNC color_type get_ce() const
    {
        return ce_;
    }

    VSNRAY_FUNC void set_ls(scalar_type const& ls)
    {
        ls_ = ls;
    }

    VSNRAY_FUNC scalar_type get_ls() const
    {
        return ls_;
    }

private:

    color_type  ce_;
    scalar_type ls_;

};


//-------------------------------------------------------------------------------------------------
// Matte material
//

template <typename T>
class matte
{
public:

    typedef T scalar_type;
    typedef vector<3, T> color_type;

    VSNRAY_FUNC color_type ambient() const
    {
        return ca_ * ka_;
    }

    template <typename L, typename U>
    VSNRAY_FUNC
    vector<3, U> shade(shade_record<L, U> const& sr) const
    {
        using V = vector<3, U>;
        V result(0.0, 0.0, 0.0);

        auto l = *sr.light;
        auto wi = sr.light_dir;
        auto wo = sr.view_dir;
        auto ndotl = dot(sr.normal, wi);

        auto mask = sr.active & (ndotl > U(0.0));
        auto c = constants::pi<U>() * diffuse_brdf_.f(sr.normal, wo, wi) * V(l.color()) * V(ndotl);
        result = add( result, c, mask );

        return result;
    }

    template <typename SR, typename U, typename S /* sampler */>
    VSNRAY_FUNC
    vector<3, U> sample(SR const& shade_rec, vector<3, U>& refl_dir, U& pdf, S& sampler)
    {
        return diffuse_brdf_.sample_f(shade_rec.normal, shade_rec.view_dir, refl_dir, pdf, sampler);
    }


    VSNRAY_FUNC void set_ca(color_type const& ca)
    {
        ca_ = ca;
    }

    VSNRAY_FUNC void set_ka(scalar_type ka)
    {
        ka_ = ka;
    }

    VSNRAY_FUNC void set_cd(color_type const& cd)
    {
        diffuse_brdf_.cd = cd;
    }

    VSNRAY_FUNC void set_kd(scalar_type kd)
    {
        diffuse_brdf_.kd = kd;
    }

private:

    color_type      ca_;
    scalar_type     ka_;
    lambertian<T>   diffuse_brdf_;

};


//-------------------------------------------------------------------------------------------------
// Plastic material
//

template <typename T>
class plastic
{
public:

    typedef T scalar_type;
    typedef vector<3, T> color_type;

    VSNRAY_FUNC color_type ambient() const
    {
        return ca_ * ka_;
    }

    template <typename SR>
    VSNRAY_FUNC
    vector<3, typename SR::scalar_type> shade(SR const& sr) const
    {
        using U = typename SR::scalar_type;
        using V = vector<3, U>;

        auto l = *sr.light;
        auto wi = sr.light_dir;
        auto wo = sr.view_dir;
        auto n = sr.normal;
#if 1 // two-sided
        n = select( dot(n, wo) < U(0.0), -n, n );
#endif
        auto ndotl = max( U(0.0), dot(n, wi) );

        U att(1.0);

#if 1 // use attenuation
        auto dist = length(V(l.position()) - sr.isect_pos);
        att = U(
                1.0 / (l.constant_attenuation()
                     + l.linear_attenuation() * dist
                     + l.quadratic_attenuation() * dist * dist)
            );
#endif

        return constants::pi<U>() * ( cd(sr, n, wo, wi) + specular_brdf_.f(n, wo, wi) ) * att * V(l.color()) * V(ndotl);
    }

    template <typename SR, typename U, typename S /* sampler */>
    VSNRAY_FUNC
    vector<3, U> sample(SR const& shade_rec, vector<3, U>& refl_dir, U& pdf, S& sampler)
    {
        return diffuse_brdf_.sample_f(shade_rec.normal, shade_rec.view_dir, refl_dir, pdf, sampler);
    }

    template <typename L, typename C, typename U, typename S /* sampler */>
    VSNRAY_FUNC
    vector<3, U> sample(shade_record<L, C, U> const& sr, vector<3, U>& refl_dir, U& pdf, S& sampler)
    {
        return vector<3, U>(sr.cd) * diffuse_brdf_.sample_f(sr.normal, sr.view_dir, refl_dir, pdf, sampler);
    }


    VSNRAY_FUNC void set_ca(color_type const& ca)
    {
        ca_ = ca;
    }

    VSNRAY_FUNC color_type get_ca() const
    {
        return ca_;
    }

    VSNRAY_FUNC void set_ka(scalar_type ka)
    {
        ka_ = ka;
    }

    VSNRAY_FUNC scalar_type get_ka() const
    {
        return ka_;
    }

    VSNRAY_FUNC void set_cd(color_type const& cd)
    {
        diffuse_brdf_.cd = cd;
    }

    VSNRAY_FUNC color_type get_cd() const
    {
        return diffuse_brdf_.cd;
    }

    VSNRAY_FUNC void set_kd(scalar_type kd)
    {
        diffuse_brdf_.kd = kd;
    }

    VSNRAY_FUNC scalar_type get_kd() const
    {
        return diffuse_brdf_.kd;
    }

    VSNRAY_FUNC void set_cs(color_type const& cs)
    {
        specular_brdf_.cs = cs;
    }

    VSNRAY_FUNC color_type get_cs() const
    {
        return specular_brdf_.cs;
    }

    VSNRAY_FUNC void set_ks(scalar_type ks)
    {
        specular_brdf_.ks = ks;
    }

    VSNRAY_FUNC scalar_type get_ks() const
    {
        return specular_brdf_.ks;
    }

    VSNRAY_FUNC void set_specular_exp(scalar_type exp)
    {
        specular_brdf_.exp = exp;
    }

    VSNRAY_FUNC scalar_type get_specular_exp() const
    {
        return specular_brdf_.exp;
    }

private:

    color_type      ca_;
    scalar_type     ka_;
    lambertian<T>   diffuse_brdf_;
    phong<T>        specular_brdf_;

    template <typename SR, typename V>
    VSNRAY_FUNC
    V cd(SR const& sr, V const& n, V const& wo, V const& wi) const
    {
        VSNRAY_UNUSED(sr);
        return diffuse_brdf_.f(n, wo, wi);
    }

    template <typename L, typename C, typename S, typename V>
    VSNRAY_FUNC
    V cd(shade_record<L, C, S> const& sr, V const& n, V const& wo, V const& wi) const
    {
        return V(sr.cd) * diffuse_brdf_.f(n, wo, wi);
    }

};


//-------------------------------------------------------------------------------------------------
// Generic material
//

namespace detail
{
static unsigned const EmissiveMaterial  = 0x00;
static unsigned const MatteMaterial     = 0x01;
static unsigned const PlasticMaterial   = 0x02;
} // detail


template <typename T>
struct generic_mat
{
    typedef T scalar_type;

    unsigned type_;

    generic_mat()
    {
    }

    /* implicit */ generic_mat(emissive<T> const& e)
        : type_(detail::EmissiveMaterial)
        , emissive_mat(e)
    {
    }

    /* implicit */ generic_mat(matte<T> const& m)
        : type_(detail::MatteMaterial)
        , matte_mat(m)
    {
    }

    /* implicit */ generic_mat(plastic<T> const& p)
        : type_(detail::PlasticMaterial)
        , plastic_mat(p)
    {
    }

    void operator=(emissive<T> const& e)
    {
        type_ = detail::EmissiveMaterial;
        emissive_mat = e;
    }

    void operator=(matte<T> const& m)
    {
        type_ = detail::MatteMaterial;
        matte_mat = m;
    }

    void operator=(plastic<T> const& p)
    {
        type_ = detail::PlasticMaterial;
        plastic_mat = p;
    }

    VSNRAY_FUNC
    unsigned get_type() const
    {
        return type_;
    }

    template <typename L, typename U>
    VSNRAY_FUNC
    vector<3, U> shade(shade_record<L, U> const& sr) const
    {
        switch (type_)
        {
        case detail::EmissiveMaterial:
            return emissive_mat.shade(sr);

        case detail::MatteMaterial:
            return matte_mat.shade(sr);

        case detail::PlasticMaterial:
            return plastic_mat.shade(sr);
        }

        VSNRAY_UNREACHABLE();
    }

    template <typename SR, typename U, typename S /* sampler */>
    VSNRAY_FUNC
    vector<3, U> sample(SR const& shade_rec, vector<3, U>& refl_dir, U& pdf, S& sampler)
    {
        switch (type_)
        {
        case detail::EmissiveMaterial:
            return emissive_mat.sample(shade_rec, refl_dir, pdf, sampler);

        case detail::MatteMaterial:
            return matte_mat.sample(shade_rec, refl_dir, pdf, sampler);

        case detail::PlasticMaterial:
            return plastic_mat.sample(shade_rec, refl_dir, pdf, sampler);
        }

        VSNRAY_UNREACHABLE();
    }


    union
    {
        emissive<T>     emissive_mat;
        matte<T>        matte_mat;
        plastic<T>      plastic_mat;
    };
};

template <>
class generic_mat<simd::float4>
{
public:

    typedef simd::float4 scalar_type;

    generic_mat
    (
        generic_mat<float> const& m1, generic_mat<float> const& m2,
        generic_mat<float> const& m3, generic_mat<float> const& m4
    )
        : m1_(m1)
        , m2_(m2)
        , m3_(m3)
        , m4_(m4)
        , type_( simd::int4(m1.get_type(), m2.get_type(), m3.get_type(), m4.get_type()) )
    {
    }

    simd::int4 get_type() const
    {
        return type_;
    }

    template <typename L>
    vector<3, simd::float4> shade(shade_record<L, simd::float4> const& sr) const
    {
        auto sr4 = simd::unpack(sr);
        vector<3, float> v[] =
        {
            vector<3, float>( m1_.shade(sr4[0]) ),
            vector<3, float>( m2_.shade(sr4[1]) ),
            vector<3, float>( m3_.shade(sr4[2]) ),
            vector<3, float>( m4_.shade(sr4[3]) )
        };
        return simd::pack( v[0], v[1], v[2], v[3] );
    }

    template <typename L, typename S /* sampler */>
    vector<3, simd::float4> sample(shade_record<L, simd::float4> const& sr, vector<3, simd::float4>& refl_dir, simd::float4& pdf, S& samp)
    {
        auto sr4 = simd::unpack(sr);
        vector<3, float> rd4[4];
        VSNRAY_ALIGN(16) float pdf4[] = { 0.0f, 0.0f, 0.0f, 0.0f };
        auto& s = samp.get_sampler();
        vector<3, float> v[] =
        {
            vector<3, float>( m1_.sample(sr4[0], rd4[0], pdf4[0], s) ),
            vector<3, float>( m2_.sample(sr4[1], rd4[1], pdf4[1], s) ),
            vector<3, float>( m3_.sample(sr4[2], rd4[2], pdf4[2], s) ),
            vector<3, float>( m4_.sample(sr4[3], rd4[3], pdf4[3], s) )
        };
        refl_dir = simd::pack( rd4[0], rd4[1], rd4[2], rd4[3] );
        pdf = simd::float4(pdf4);
        return simd::pack( v[0], v[1], v[2], v[3] );
    }

private:

    generic_mat<float> m1_;
    generic_mat<float> m2_;
    generic_mat<float> m3_;
    generic_mat<float> m4_;

    simd::int4 type_;

};


} // visionaray

#include "detail/material.inl"

#endif // VSNRAY_MATERIAL_H
