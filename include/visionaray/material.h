// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_MATERIAL_H
#define VSNRAY_MATERIAL_H

#include "detail/macros.h"
#include "math/math.h"
#include "brdf.h"
#include "shade_record.h"
#include "spectrum.h"
#include "variant.h"


namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Emissive material, no BRDFs
//

template <typename T>
class emissive
{
public:

    using scalar_type   = T;

public:

    VSNRAY_FUNC spectrum<T> ambient() const
    {
        return spectrum<T>();
    }

    template <typename SR, typename U, typename S /* sampler */>
    VSNRAY_FUNC
    spectrum<U> sample(SR const& shade_rec, vector<3, U>& refl_dir, U& pdf, S& sampler) const
    {
        VSNRAY_UNUSED(refl_dir); // TODO?
        VSNRAY_UNUSED(sampler);
        pdf = U(1.0);
        return shade(shade_rec);
    }

    template <typename SR>
    VSNRAY_FUNC
    spectrum<typename SR::scalar_type> shade(SR const& sr) const
    {
        using U = typename SR::scalar_type;
        return select( dot(sr.normal, sr.view_dir) >= U(0.0), ce(sr), spectrum<U>(0.0) );
    }


    VSNRAY_FUNC void set_ce(spectrum<T> const& ce)
    {
        ce_ = ce;
    }

    VSNRAY_FUNC spectrum<T> get_ce() const
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

    spectrum<T> ce_;
    scalar_type ls_;

    template <typename SR>
    VSNRAY_FUNC
    spectrum<T> ce(SR const& sr) const
    {
        VSNRAY_UNUSED(sr);
        return ce_ * ls_;
    }

    template <typename L, typename C, typename S>
    VSNRAY_FUNC
    spectrum<T> ce(shade_record<L, C, S> const& sr) const
    {
        return spectrum<T>(from_rgb(sr.tex_color)) * ce_ * ls_;
    }

};


//-------------------------------------------------------------------------------------------------
// Matte material
//

template <typename T>
class matte
{
public:

    using scalar_type   = T;

public:

    VSNRAY_FUNC spectrum<T> ambient() const
    {
        return ca_ * ka_;
    }

    template <typename SR>
    VSNRAY_FUNC
    spectrum<typename SR::scalar_type> shade(SR const& sr) const
    {
        using U = typename SR::scalar_type;
        using V = vector<3, U>;

        spectrum<U> result(0.0);

        auto l = sr.light;
        auto wi = sr.light_dir;
        auto wo = sr.view_dir;
        auto n = sr.normal;
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

        return spectrum<U>(
                constants::pi<U>()
              * cd(sr, n, wo, wi)
              * spectrum<U>(from_rgb(l.color()))
              * ndotl
              * att
                );
    }

    template <typename SR, typename U, typename S /* sampler */>
    VSNRAY_FUNC
    spectrum<U> sample(SR const& shade_rec, vector<3, U>& refl_dir, U& pdf, S& sampler) const
    {
        return sample_impl(shade_rec, refl_dir, pdf, sampler);
    }

    template <typename L, typename C, typename U, typename S /* sampler */>
    VSNRAY_FUNC
    spectrum<U> sample(shade_record<L, C, U> const& sr, vector<3, U>& refl_dir, U& pdf, S& sampler) const
    {
        return spectrum<U>(from_rgb(sr.tex_color)) * sample_impl(sr, refl_dir, pdf, sampler);
    }


    VSNRAY_FUNC void set_ca(spectrum<T> const& ca)
    {
        ca_ = ca;
    }

    VSNRAY_FUNC void set_ka(scalar_type ka)
    {
        ka_ = ka;
    }

    VSNRAY_FUNC void set_cd(spectrum<T> const& cd)
    {
        diffuse_brdf_.cd = cd;
    }

    VSNRAY_FUNC void set_kd(scalar_type kd)
    {
        diffuse_brdf_.kd = kd;
    }

private:

    spectrum<T>     ca_;
    scalar_type     ka_;
    lambertian<T>   diffuse_brdf_;

    template <typename SR, typename V>
    VSNRAY_FUNC
    spectrum<T> cd(SR const& sr, V const& n, V const& wo, V const& wi) const
    {
        VSNRAY_UNUSED(sr);
        return diffuse_brdf_.f(n, wo, wi);
    }

    template <typename L, typename C, typename S, typename V>
    VSNRAY_FUNC
    spectrum<T> cd(shade_record<L, C, S> const& sr, V const& n, V const& wo, V const& wi) const
    {
        return spectrum<T>(from_rgb(sr.tex_color)) * diffuse_brdf_.f(n, wo, wi);
    }

    template <typename SR, typename U, typename S /* sampler */>
    VSNRAY_FUNC
    spectrum<U> sample_impl(SR const& shade_rec, vector<3, U>& refl_dir, U& pdf, S& sampler) const
    {
        return diffuse_brdf_.sample_f(shade_rec.normal, shade_rec.view_dir, refl_dir, pdf, sampler);
    }
};


//-------------------------------------------------------------------------------------------------
// Plastic material
//

template <typename T>
class plastic
{
public:

    using scalar_type   = T;

public:

    VSNRAY_FUNC spectrum<T> ambient() const
    {
        return ca_ * ka_;
    }

    template <typename SR>
    VSNRAY_FUNC
    spectrum<typename SR::scalar_type> shade(SR const& sr) const
    {
        using U = typename SR::scalar_type;
        using V = vector<3, U>;

        auto l = sr.light;
        auto wi = sr.light_dir;
        auto wo = sr.view_dir;
        auto n = sr.normal;
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

        return spectrum<U>(
                constants::pi<U>()
              * ( cd(sr, n, wo, wi) + specular_brdf_.f(n, wo, wi) )
              * spectrum<U>(from_rgb(l.color()))
              * ndotl
              * att
                );
    }

    template <typename SR, typename U, typename S /* sampler */>
    VSNRAY_FUNC
    spectrum<U> sample(SR const& shade_rec, vector<3, U>& refl_dir, U& pdf, S& sampler) const
    {
        return sample_impl(shade_rec, refl_dir, pdf, sampler);
    }

    template <typename L, typename C, typename U, typename S /* sampler */>
    VSNRAY_FUNC
    spectrum<U> sample(shade_record<L, C, U> const& sr, vector<3, U>& refl_dir, U& pdf, S& sampler) const
    {
        return spectrum<U>(from_rgb(sr.tex_color)) * sample_impl(sr, refl_dir, pdf, sampler);
    }


    VSNRAY_FUNC void set_ca(spectrum<T> const& ca)
    {
        ca_ = ca;
    }

    VSNRAY_FUNC spectrum<T> get_ca() const
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

    VSNRAY_FUNC void set_cd(spectrum<T> const& cd)
    {
        diffuse_brdf_.cd = cd;
    }

    VSNRAY_FUNC spectrum<T> get_cd() const
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

    VSNRAY_FUNC void set_cs(spectrum<T> const& cs)
    {
        specular_brdf_.cs = cs;
    }

    VSNRAY_FUNC spectrum<T> get_cs() const
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

    spectrum<T>     ca_;
    scalar_type     ka_;
    lambertian<T>   diffuse_brdf_;
    blinn<T>        specular_brdf_;

    template <typename SR, typename V>
    VSNRAY_FUNC
    spectrum<T> cd(SR const& sr, V const& n, V const& wo, V const& wi) const
    {
        VSNRAY_UNUSED(sr);
        return diffuse_brdf_.f(n, wo, wi);
    }

    template <typename L, typename C, typename S, typename V>
    VSNRAY_FUNC
    spectrum<T> cd(shade_record<L, C, S> const& sr, V const& n, V const& wo, V const& wi) const
    {
        return spectrum<T>(from_rgb(sr.tex_color)) * diffuse_brdf_.f(n, wo, wi);
    }

    template <typename SR, typename U, typename S /* sampler */>
    VSNRAY_FUNC
    spectrum<U> sample_impl(SR const& sr, vector<3, U>& refl_dir, U& pdf, S& sampler) const
    {
        U pdf1;
        U pdf2;

        vector<3, U> refl1;
        vector<3, U> refl2;

        spectrum<U>  diff;
        spectrum<U>  spec;

        auto prob_diff = mean_value( diffuse_brdf_.cd ) * diffuse_brdf_.kd;
        auto prob_spec = mean_value( specular_brdf_.cs ) * specular_brdf_.ks;

        auto all_zero  = prob_diff == U(0.0) && prob_spec == U(0.0);

        prob_diff      = select( all_zero, U(0.5), prob_diff );
        prob_spec      = select( all_zero, U(0.5), prob_spec );

        prob_diff      = prob_diff / (prob_diff + prob_spec);


        auto u         = sampler.next();

        if ( any(sr.active && u < U(prob_diff)) )
        {
            diff       = diffuse_brdf_.sample_f(sr.normal, sr.view_dir, refl1, pdf1, sampler);
        }

        if ( any(sr.active && u >= U(prob_diff)) )
        {
            spec       = specular_brdf_.sample_f(sr.normal, sr.view_dir, refl2, pdf2, sampler);
        }

        pdf            = select( u < U(prob_diff), pdf1,  pdf2  );
        refl_dir       = select( u < U(prob_diff), refl1, refl2 );

        return           select( u < U(prob_diff), diff,  spec  );
    }

};

} // visionaray

#include "detail/material.inl"

#endif // VSNRAY_MATERIAL_H
