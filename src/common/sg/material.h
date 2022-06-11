// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_SG_MATERIAL_H
#define VSNRAY_COMMON_SG_MATERIAL_H 1

#include <string>

#include <visionaray/math/forward.h>
#include <visionaray/math/vector.h>

namespace visionaray
{
namespace sg
{

//-------------------------------------------------------------------------------------------------
// Material base class
//

class material
{
public:
    virtual ~material() {}

    std::string& name();
    std::string const& name() const;

private:

    std::string name_;
};


//-------------------------------------------------------------------------------------------------
// Wavefront obj material
//

struct obj_material : material
{
    // Ambient color
    vec3 ca = { 0.2f, 0.2f, 0.2f };

    // Diffuse color
    vec3 cd = { 0.8f, 0.8f, 0.8f };

    // Specular color
    vec3 cs = { 0.1f, 0.1f, 0.1f };

    // Emissive color
    vec3 ce = { 0.0f, 0.0f, 0.0f };

    // Reflective color
    vec3 cr = { 0.0f, 0.0f, 0.0f };

    // Index of refraction
    vec3 ior = { 1.0f, 1.0f, 1.0f };

    // Absorption
    vec3 absorption = { 0.0f, 0.0f, 0.0f };

    // Transmission
    float transmission = 0.0f;

    // Specular exponent
    float specular_exp = 32.0f;

    // Wavefront obj illumination model (default: 1 maps to plastic).
    int illum = 2;
};


//-------------------------------------------------------------------------------------------------
// Glass material
//

struct glass_material : material
{
    // Transmissive color
    vec3 ct = { 0.8f, 0.8f, 0.8f };

    // Reflective color
    vec3 cr = { 0.2f, 0.2f, 0.2f };

    // Index of refraction
    vec3 ior = { 1.0f, 1.0f, 1.0f };
};


//-------------------------------------------------------------------------------------------------
// Metal material
//

struct metal_material : material
{
    // Roughness
    float roughness = 0.0f;

    // Index of refraction
    vec3 ior = { 1.0f, 1.0f, 1.0f };

    // Absorption
    vec3 absorption = { 0.2f, 0.2f, 0.2f };
};


//-------------------------------------------------------------------------------------------------
// Disney principled material
//

struct disney_material : material
{
    // Base color
    vec4 base_color = vec4(0.0f);

    // Specular transmission
    float spec_trans = 0.0f;

    // Sheen
    float sheen = 0.0f;

    // Sheen tint
    float sheen_tint = 0.0f;

    // Index of refraction
    float ior = 1.0f;

    // Refractivity
    float refractive = 0.0f;

    // Roughness
    float roughness = 0.0f;


    // TODO..
};

} // sg
} // visionaray

#endif // VSNRAY_COMMON_SG_MATERIAL_H
