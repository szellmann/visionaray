// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_MATERIAL_H
#define VSNRAY_MATERIAL_H

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

    template <typename L, typename U>
    VSNRAY_FUNC
    vector<3, U> shade(shade_record<L, U> const& sr) const
    {
        vector<3, U> result(0.0, 0.0, 0.0);

        auto l = *sr.light;
        auto wi = normalize(vector<3, U>(l.position_));
        auto wo = sr.view_dir;
        auto ndotl = dot(sr.normal, wi);

        auto mask = sr.active & (ndotl >= U(0.0));
        auto c = mul( diffuse_brdf_.f(sr.normal, wo, wi), vector<3, U>(ndotl), mask );
        result = add( result, c, mask );

        return result;
    }

    template <typename SR, typename U, typename S /* sampler */>
    VSNRAY_FUNC
    vector<3, U> sample(SR const& shade_rec, vector<3, U>& refl_dir, U& pdf, S& sampler)
    {
        return diffuse_brdf_.sample_f(shade_rec.normal, shade_rec.view_dir, refl_dir, pdf, sampler);
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

    lambertian<T> diffuse_brdf_;

};


//-------------------------------------------------------------------------------------------------
// Phong material
//

template <typename T>
class phong
{
public:

    typedef T scalar_type;
    typedef vector<3, T> color_type;

    template <typename SR>
    VSNRAY_FUNC
    vector<3, typename SR::scalar_type> shade(SR const& sr) const
    {
        typedef typename SR::scalar_type    scalar_type_in;
        typedef vector<3, scalar_type_in>   vec_type_in;

        vec_type_in result(0.0, 0.0, 0.0);

        auto l = *sr.light;
        auto wi = normalize(vec_type_in(l.position_));
        auto wo = sr.view_dir;
        auto ndotl = dot(sr.normal, wi);

        auto mask = sr.active & (ndotl >= scalar_type_in(0.0));
        auto c = mul( add(cd(sr, wo, wi), specular_brdf_.f(sr.normal, wo, wi), mask), vec_type_in(ndotl), mask );
        result = add( result, c, mask );

        return result;
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

    lambertian<T>   diffuse_brdf_;
    specular<T>     specular_brdf_;

    template <typename SR, typename VecT>
    VSNRAY_FUNC
    VecT cd(SR const& sr, VecT const& wo, VecT const& wi) const
    {
        return diffuse_brdf_.f(sr.normal, wo, wi);
    }

    template <typename L, typename C, typename VecT>
    VSNRAY_FUNC
    VecT cd(shade_record<L, C, typename VecT::value_type> const& sr, VecT const& wo, VecT const& wi) const
    {
        return VecT(sr.cd) * diffuse_brdf_.f(sr.normal, wo, wi);
    }

};


//-------------------------------------------------------------------------------------------------
// Generic material
//

namespace detail
{
static unsigned const EmissiveMaterial  = 0x00;
static unsigned const MatteMaterial     = 0x01;
static unsigned const PhongMaterial     = 0x02;
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

    /* implicit */ generic_mat(phong<T> const& p)
        : type_(detail::PhongMaterial)
        , phong_mat(p)
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

    void operator=(phong<T> const& p)
    {
        type_ = detail::PhongMaterial;
        phong_mat = p;
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

        case detail::PhongMaterial:
            return phong_mat.shade(sr);
        }
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

        case detail::PhongMaterial:
            return phong_mat.sample(shade_rec, refl_dir, pdf, sampler);
        }
    }


    union
    {
        emissive<T>     emissive_mat;
        matte<T>        matte_mat;
        phong<T>        phong_mat;
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
    {
    }

    template <typename L>
    VSNRAY_FUNC
    vector<3, simd::float4> shade(shade_record<L, simd::float4> const& sr) const
    {
        auto sr4 = unpack(sr);
        auto v{ m1_.shade(sr4[0]), m2_.shade(sr4[1]), m3_.shade(sr4[2]), m4_.shade(sr4[3]) };
        return pack( v[0], v[1], v[2], v[3] );
    }

    template <typename L, typename S /* sampler */>
    VSNRAY_FUNC
    vector<3, simd::float4> sample(shade_record<L, simd::float4> const& sr, vector<3, simd::float4>& refl_dir, simd::float4& pdf, S& sampler)
    {
/*        auto sr4 = unpack(sr);
        vector<3, float> rd4;
        VSNRAY_ALIGN float pdf4[4];*/
    }

private:

    generic_mat<float> m1_;
    generic_mat<float> m2_;
    generic_mat<float> m3_;
    generic_mat<float> m4_;

};


} // visionaray

#include "detail/material.inl"

#endif // VSNRAY_MATERIAL_H


