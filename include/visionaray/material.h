// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_MATERIAL_H
#define VSNRAY_MATERIAL_H 1

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

    using scalar_type = T;

public:

    VSNRAY_FUNC spectrum<T> ambient() const;

    template <typename SR>
    VSNRAY_FUNC spectrum<typename SR::scalar_type> shade(SR const& sr) const;

    template <typename SR, typename U, typename Sampler>
    VSNRAY_FUNC spectrum<U> sample(
            SR const&       shade_rec,
            vector<3, U>&   refl_dir,
            U&              pdf,
            Sampler&        sampler) const;

    VSNRAY_FUNC void set_ce(spectrum<T> const& ce);
    VSNRAY_FUNC spectrum<T> get_ce() const;

    VSNRAY_FUNC void set_ls(T ls);
    VSNRAY_FUNC T get_ls() const;

private:

    spectrum<T> ce_;
    T           ls_;

    template <typename SR>
    VSNRAY_FUNC spectrum<T> ce(SR const& sr) const;

    template <typename L, typename C, typename S>
    VSNRAY_FUNC spectrum<T> ce(shade_record<L, C, S> const& sr) const;

};


//-------------------------------------------------------------------------------------------------
// Matte material
//

template <typename T>
class matte
{
public:

    using scalar_type = T;

public:

    VSNRAY_FUNC spectrum<T> ambient() const;

    template <typename SR>
    VSNRAY_FUNC
    spectrum<typename SR::scalar_type> shade(SR const& sr) const;

    template <typename SR, typename U, typename Sampler>
    VSNRAY_FUNC spectrum<U> sample(
            SR const&       shade_rec,
            vector<3, U>&   refl_dir,
            U&              pdf,
            Sampler&        sampler
            ) const;

    template <typename L, typename C, typename U, typename Sampler>
    VSNRAY_FUNC spectrum<U> sample(
            shade_record<L, C, U> const&    sr,
            vector<3, U>&                   refl_dir,
            U&                              pdf,
            Sampler&                        sampler
            ) const;


    VSNRAY_FUNC void set_ca(spectrum<T> const& ca);
    VSNRAY_FUNC spectrum<T> get_ca() const;

    VSNRAY_FUNC void set_ka(T ka);
    VSNRAY_FUNC T get_ka() const;

    VSNRAY_FUNC void set_cd(spectrum<T> const& cd);
    VSNRAY_FUNC spectrum<T> get_cd() const;

    VSNRAY_FUNC void set_kd(T kd);
    VSNRAY_FUNC T get_kd() const;

private:

    spectrum<T>     ca_;
    T               ka_;
    lambertian<T>   diffuse_brdf_;

    template <typename SR, typename V>
    VSNRAY_FUNC spectrum<T> cd(SR const& sr, V const& n, V const& wo, V const& wi) const;

    template <typename L, typename C, typename S, typename V>
    VSNRAY_FUNC spectrum<T> cd(shade_record<L, C, S> const& sr, V const& n, V const& wo, V const& wi) const;

    template <typename SR, typename U, typename Sampler>
    VSNRAY_FUNC
    spectrum<U> sample_impl(
            SR const&       shade_rec,
            vector<3, U>&   refl_dir,
            U&              pdf,
            Sampler&        sampler
            ) const;
};


//-------------------------------------------------------------------------------------------------
// Mirror material
//

template <typename T>
class mirror
{
public:

    using scalar_type = T;

public:

    // TODO: no support for  ambient (function returns 0.0)
    VSNRAY_FUNC spectrum<T> ambient() const;

    template <typename SR>
    VSNRAY_FUNC
    spectrum<typename SR::scalar_type> shade(SR const& sr) const;

    template <typename SR, typename U, typename Sampler>
    VSNRAY_FUNC spectrum<U> sample(
            SR const&       sr,
            vector<3, U>&   refl_dir,
            U&              pdf,
            Sampler&        sampler
            ) const;

    VSNRAY_FUNC void set_cr(spectrum<T> const& cr);
    VSNRAY_FUNC spectrum<T> get_cr() const;

    VSNRAY_FUNC void set_kr(T const& kr);
    VSNRAY_FUNC T get_kr() const;

    VSNRAY_FUNC void set_ior(spectrum<T> const& ior);
    VSNRAY_FUNC void set_ior(T ior);
    VSNRAY_FUNC spectrum<T> get_ior() const;

    VSNRAY_FUNC void set_absorption(spectrum<T> const& absorption);
    VSNRAY_FUNC void set_absorption(T absorption);
    VSNRAY_FUNC spectrum<T> get_absorption() const;

private:

    specular_reflection<T>  specular_brdf_;

};


//-------------------------------------------------------------------------------------------------
// Plastic material
//

template <typename T>
class plastic
{
public:

    using scalar_type = T;

public:

    VSNRAY_FUNC spectrum<T> ambient() const;

    template <typename SR>
    VSNRAY_FUNC
    spectrum<typename SR::scalar_type> shade(SR const& sr) const;

    template <typename SR, typename U, typename Sampler>
    VSNRAY_FUNC spectrum<U> sample(
            SR const&       shade_rec,
            vector<3, U>&   refl_dir,
            U&              pdf,
            Sampler&        sampler
            ) const;

    template <typename L, typename C, typename U, typename Sampler>
    VSNRAY_FUNC spectrum<U> sample(
            shade_record<L, C, U> const&    sr,
            vector<3, U>&                   refl_dir,
            U&                              pdf,
            Sampler&                        sampler
            ) const;

    VSNRAY_FUNC void set_ca(spectrum<T> const& ca);
    VSNRAY_FUNC spectrum<T> get_ca() const;

    VSNRAY_FUNC void set_ka(T ka);
    VSNRAY_FUNC T get_ka() const;

    VSNRAY_FUNC void set_cd(spectrum<T> const& cd);
    VSNRAY_FUNC spectrum<T> get_cd() const;

    VSNRAY_FUNC void set_kd(T kd);
    VSNRAY_FUNC T get_kd() const;

    VSNRAY_FUNC void set_cs(spectrum<T> const& cs);
    VSNRAY_FUNC spectrum<T> get_cs() const;

    VSNRAY_FUNC void set_ks(T ks);
    VSNRAY_FUNC T get_ks() const;

    VSNRAY_FUNC void set_specular_exp(T exp);
    VSNRAY_FUNC T get_specular_exp() const;

private:

    spectrum<T>     ca_;
    T               ka_;
    lambertian<T>   diffuse_brdf_;
    blinn<T>        specular_brdf_;

    template <typename SR, typename V>
    VSNRAY_FUNC spectrum<T> cd(SR const& sr, V const& n, V const& wo, V const& wi) const;

    template <typename L, typename C, typename S, typename V>
    VSNRAY_FUNC spectrum<T> cd(shade_record<L, C, S> const& sr, V const& n, V const& wo, V const& wi) const;

    template <typename SR, typename U, typename Sampler>
    VSNRAY_FUNC spectrum<U> sample_impl(
            SR const&       sr,
            vector<3, U>&   refl_dir,
            U&              pdf,
            Sampler&        sampler
            ) const;

};

} // visionaray

#include "detail/material/emissive.inl"
#include "detail/material/matte.inl"
#include "detail/material/mirror.inl"
#include "detail/material/plastic.inl"
#include "detail/material.inl"

#endif // VSNRAY_MATERIAL_H
