// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_MAKE_MATERIALS_H
#define VSNRAY_COMMON_MAKE_MATERIALS_H 1

#include <visionaray/aligned_vector.h>
#include <visionaray/generic_material.h>
#include <visionaray/material.h>

#include "sg/material.h"

namespace visionaray
{

inline aligned_vector<plastic<float>> make_materials(
        plastic<float>                          /* */,
        aligned_vector<sg::obj_material> const& materials
        )
{
    aligned_vector<plastic<float>> result;

    for (auto mat : materials)
    {
        plastic<float> pl;
        pl.ca() = from_rgb(mat.ca);
        pl.cd() = from_rgb(mat.cd);
        pl.cs() = from_rgb(mat.cs);
        pl.ka() = 1.0f;
        pl.kd() = 1.0f;
        pl.ks() = 1.0f;
        pl.specular_exp() = mat.specular_exp;
        result.emplace_back(pl);
    }

    return result;
}

template <typename ...Ts, typename UpdateFunc>
inline aligned_vector<generic_material<Ts...>> make_materials(
        generic_material<Ts...>                 /* */,
        aligned_vector<sg::obj_material> const& materials,
        UpdateFunc                              update_func
        )
{
    aligned_vector<generic_material<Ts...>> result;

    for (auto mat : materials)
    {
        update_func(result, mat);
    }

    return result;
}

} // visionaray

#endif // VSNRAY_COMMON_MAKE_MATERIALS_H
