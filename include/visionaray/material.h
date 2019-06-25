// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_MATERIAL_H
#define VSNRAY_MATERIAL_H 1

#include "detail/macros.h"
#include "math/vector.h"
#include "brdf.h"
#include "shade_record.h"
#include "spectrum.h"


namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Material classes
//
//
// Built-in and user-defined materials (must) support the following interface:
//
//  - shade():
//      const parameter shade_record:   shading info (normal, texture color, ...)
//      return type:                    spectrum
//
//  - sample():
//      const parameter shade_record:   shading info (normal, texture color, ...)
//      modifiable parameter refl_dir:  outgoing light direction (computed by function,
//                                      must be normalized)
//      modifiable parameter pdf:       probability density function of the the sampled BRDF
//      modifiable parameter inter:     interaction type for tracking (e.g. to perform MIS later)
//      modifiable parameter sampler:   implements sampler interface to get pseudo random
//                                      numbers or quasi random numbers
//
//
// Built-in materials
//
//  - disney:
//      Disney's Principled BRDF. WIP!
//
//  - emissive:
//      shade() returns radiance emitted by the surface
//      mind that radiance for emitters beyond 1.0 cd/m^2 is physically plausible
//      no reflective properties, sample() will provide no valid reflection result
//
//  - matte:
//      material with only diffuse reflection properties
//
//  - mirror:
//      material with only perfectly specular reflection properties
//
//  - plastic:
//      OpenGL-like material with a diffuse BRDF (Lambertian reflection) and a specular BRDF
//      (Blinn reflection model). When being sampled, the diffuse and specular terms
//      are used to determine whether the diffuse or specular BRDF will contribute:
//
//          Pd = mean_value(cd) * kd        /* probability of diff. reflection */
//          Ps = mean_value(cs) * ks        /* probability of spec. reflection */
//
//          Pd = Pd / (Pd + Ps)             /* normalization */
//          Ps = Ps / (Pd + Ps)
//
//
// Compatibility with generic_material<Ts...>
//
//  User-defined materials are compatible with the generic_material mechanism if
// they implement the interface described above [...]
//
//  TODO: make variant-type more versatile:
//
//  [...] and if they contain no constructors other than the compiler-generated
//  constructors. The same applies do destructors and assignment operators.
//
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
// Disney material
//
// WIP!
//

template <typename T>
class disney
{
public:

    using scalar_type = T;

public:

    VSNRAY_FUNC spectrum<T> ambient() const;

    template <typename SR>
    VSNRAY_FUNC
    spectrum<typename SR::scalar_type> shade(SR const& sr) const;

    template <typename SR, typename U, typename Interaction, typename Generator>
    VSNRAY_FUNC spectrum<U> sample(
            SR const&       shade_rec,
            vector<3, U>&   refl_dir,
            U&              pdf,
            Interaction&    inter,
            Generator&      gen
            ) const;

    template <typename SR, typename Interaction>
    VSNRAY_FUNC typename SR::scalar_type pdf(
            SR const&          shared_rec,
            Interaction const& inter
            ) const;

    VSNRAY_FUNC spectrum<T>& base_color();
    VSNRAY_FUNC spectrum<T> const& base_color() const;

    VSNRAY_FUNC T& sheen();
    VSNRAY_FUNC T const& sheen() const;

    VSNRAY_FUNC T& sheen_tint();
    VSNRAY_FUNC T const& sheen_tint() const;

    VSNRAY_FUNC T& roughness();
    VSNRAY_FUNC T const& roughness() const;

private:

    disney_brdf<T> brdf_;

};


//-------------------------------------------------------------------------------------------------
// Emissive material, no BRDFs
//
// Parameters:
//  - ce: emissive spectrum
//  - ls: light intensity
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

    template <typename SR, typename U, typename Interaction, typename Generator>
    VSNRAY_FUNC spectrum<U> sample(
            SR const&       shade_rec,
            vector<3, U>&   refl_dir,
            U&              pdf,
            Interaction&    inter,
            Generator&      gen
            ) const;

    template <typename SR, typename Interaction>
    VSNRAY_FUNC typename SR::scalar_type pdf(
            SR const&          shared_rec,
            Interaction const& inter
            ) const;

    VSNRAY_FUNC spectrum<T>& ce();
    VSNRAY_FUNC spectrum<T> const& ce() const;

    VSNRAY_FUNC T& ls();
    VSNRAY_FUNC T const& ls() const;

private:

    spectrum<T> ce_;
    T           ls_;

};


//-------------------------------------------------------------------------------------------------
// Matte material
//
// Parameters:
//  - cd: diffuse spectrum
//  - kd: diffuse reflection coefficient (scales cd)
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

    template <typename SR, typename U, typename Interaction, typename Generator>
    VSNRAY_FUNC spectrum<U> sample(
            SR const&       shade_rec,
            vector<3, U>&   refl_dir,
            U&              pdf,
            Interaction&    inter,
            Generator&      gen
            ) const;

    template <typename SR, typename Interaction>
    VSNRAY_FUNC typename SR::scalar_type pdf(
            SR const&          shared_rec,
            Interaction const& inter
            ) const;

    VSNRAY_FUNC spectrum<T>& ca();
    VSNRAY_FUNC spectrum<T> const& ca() const;

    VSNRAY_FUNC T& ka();
    VSNRAY_FUNC T const& ka() const;

    VSNRAY_FUNC spectrum<T>& cd();
    VSNRAY_FUNC spectrum<T> const& cd() const;

    VSNRAY_FUNC T& kd();
    VSNRAY_FUNC T const& kd() const;

private:

    spectrum<T>     ca_;
    T               ka_;
    lambertian<T>   diffuse_brdf_;

};


//-------------------------------------------------------------------------------------------------
// Metal material
//

template <typename T>
class metal
{
public:

    using scalar_type = T;

public:

    // TODO: no support for  ambient (function returns 0.0)
    VSNRAY_FUNC spectrum<T> ambient() const;

    template <typename SR>
    VSNRAY_FUNC
    spectrum<typename SR::scalar_type> shade(SR const& sr) const;

    template <typename SR, typename U, typename Interaction, typename Generator>
    VSNRAY_FUNC spectrum<U> sample(
            SR const&       sr,
            vector<3, U>&   refl_dir,
            U&              pdf,
            Interaction&    inter,
            Generator&      gen
            ) const;

    template <typename SR, typename Interaction>
    VSNRAY_FUNC typename SR::scalar_type pdf(
            SR const&          shared_rec,
            Interaction const& inter
            ) const;

    VSNRAY_FUNC T& roughness();
    VSNRAY_FUNC T const& roughness() const;

    VSNRAY_FUNC spectrum<T>& ior();
    VSNRAY_FUNC spectrum<T> const& ior() const;

    VSNRAY_FUNC spectrum<T>& absorption();
    VSNRAY_FUNC spectrum<T> const& absorption() const;

private:

    cook_torrance<T, ggx<T>> brdf_;

};

//-------------------------------------------------------------------------------------------------
// Mirror material
//
// Parameters:
//  - cr:           perfect specular spectrum
//  - kr:           perfect specular reflection coefficient (scales cr)
//  - ior:          index of refraction for fresnel reflection
//  - absorption:   amount of radiance absorpt by the material
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

    template <typename SR, typename U, typename Interaction, typename Generator>
    VSNRAY_FUNC spectrum<U> sample(
            SR const&       sr,
            vector<3, U>&   refl_dir,
            U&              pdf,
            Interaction&    inter,
            Generator&      gen
            ) const;

    template <typename SR, typename Interaction>
    VSNRAY_FUNC typename SR::scalar_type pdf(
            SR const&          shared_rec,
            Interaction const& inter
            ) const;

    VSNRAY_FUNC spectrum<T>& cr();
    VSNRAY_FUNC spectrum<T> const& cr() const;

    VSNRAY_FUNC T& kr();
    VSNRAY_FUNC T const& kr() const;

    VSNRAY_FUNC spectrum<T>& ior();
    VSNRAY_FUNC spectrum<T> const& ior() const;

    VSNRAY_FUNC spectrum<T>& absorption();
    VSNRAY_FUNC spectrum<T> const& absorption() const;

private:

    specular_reflection<T>  specular_brdf_;

};


//-------------------------------------------------------------------------------------------------
// Glass material
//
// Parameters:
//

template <typename T>
class glass
{
public:

    using scalar_type = T;

public:

    // TODO: no support for  ambient (function returns 0.0)
    VSNRAY_FUNC spectrum<T> ambient() const;

    template <typename SR>
    VSNRAY_FUNC
    spectrum<typename SR::scalar_type> shade(SR const& sr) const;

    template <typename SR, typename U, typename Interaction, typename Generator>
    VSNRAY_FUNC spectrum<U> sample(
            SR const&       sr,
            vector<3, U>&   refl_dir,
            U&              pdf,
            Interaction&    inter,
            Generator&      gen
            ) const;

    template <typename SR, typename Interaction>
    VSNRAY_FUNC typename SR::scalar_type pdf(
            SR const&          shared_rec,
            Interaction const& inter
            ) const;

    VSNRAY_FUNC spectrum<T>& ct();
    VSNRAY_FUNC spectrum<T> const& ct() const;

    VSNRAY_FUNC T& kt();
    VSNRAY_FUNC T const& kt() const;

    VSNRAY_FUNC spectrum<T>& cr();
    VSNRAY_FUNC spectrum<T> const& cr() const;

    VSNRAY_FUNC T& kr();
    VSNRAY_FUNC T const& kr() const;

    VSNRAY_FUNC spectrum<T>& ior();
    VSNRAY_FUNC spectrum<T> const& ior() const;

private:

    specular_transmission<T>  specular_bsdf_;

};


//-------------------------------------------------------------------------------------------------
// Plastic material
//
// Parameters:
//  - cd:           diffuse spectrum
//  - kd:           diffuse reflection coefficient (scales cd)
//  - cs:           specular spectrum
//  - ks:           specular reflection coefficient (scales ks)
//  - specular_exp: controls the cone in which light may be reflected, range: [0..inf]
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

    template <typename SR, typename U, typename Interaction, typename Generator>
    VSNRAY_FUNC spectrum<U> sample(
            SR const&       shade_rec,
            vector<3, U>&   refl_dir,
            U&              pdf,
            Interaction&    inter,
            Generator&      gen
            ) const;

    template <typename SR, typename Interaction>
    VSNRAY_FUNC typename SR::scalar_type pdf(
            SR const&          shared_rec,
            Interaction const& inter
            ) const;

    VSNRAY_FUNC spectrum<T>& ca();
    VSNRAY_FUNC spectrum<T> const& ca() const;

    VSNRAY_FUNC T& ka();
    VSNRAY_FUNC T const& ka() const;

    VSNRAY_FUNC spectrum<T>& cd();
    VSNRAY_FUNC spectrum<T> const& cd() const;

    VSNRAY_FUNC T& kd();
    VSNRAY_FUNC T const& kd() const;

    VSNRAY_FUNC spectrum<T>& cs();
    VSNRAY_FUNC spectrum<T> const& cs() const;

    VSNRAY_FUNC T& ks();
    VSNRAY_FUNC T const& ks() const;

    VSNRAY_FUNC T& specular_exp();
    VSNRAY_FUNC T const& specular_exp() const;

private:

    spectrum<T>     ca_;
    T               ka_;
    lambertian<T>   diffuse_brdf_;
    blinn<T>        specular_brdf_;

};

} // visionaray

#include "detail/material/disney.inl"
#include "detail/material/emissive.inl"
#include "detail/material/glass.inl"
#include "detail/material/matte.inl"
#include "detail/material/metal.inl"
#include "detail/material/mirror.inl"
#include "detail/material/plastic.inl"
#include "detail/material.inl"

#endif // VSNRAY_MATERIAL_H
