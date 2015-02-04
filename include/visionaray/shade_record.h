// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_SHADE_RECORD_H
#define VSNRAY_SHADE_RECORD_H

#include <array>

#include "detail/tags.h"
#include "math/math.h"

namespace visionaray
{

template <typename L, typename T, typename ...Args>
struct shade_record_base
{
    typedef T scalar_type;

    vector<3, T> normal;
    vector<3, T> view_dir;
    L light;
};

template <typename L, typename T, typename ...Args>
struct shade_record : public shade_record_base<L, T, Args...>
{
    bool active;
};

template <typename L, typename C, typename T, typename ...Args>
struct shade_record<L, C, T, Args...> : public shade_record_base<L, T, Args...>
{
    C cd;
    bool active;
};

template <typename L, typename ...Args>
struct shade_record<L, simd::float4, Args...> : public shade_record_base<L, simd::float4, Args...>
{
    simd::mask4 active;
};

template <typename L, typename C, typename ...Args>
struct shade_record<L, C, simd::float4, Args...> : public shade_record_base<L, simd::float4, Args...>
{
    C cd;
    simd::mask4 active;
};


template <typename L>
std::array<shade_record<L, float>, 4> unpack(shade_record<L, simd::float4> const& sr)
{

    auto n4     = unpack(sr.normal);
    auto vd4    = unpack(sr.view_dir);
    VSNRAY_ALIGN(16) int active[4];
    store(active, sr.active);

    std::array<shade_record<L, float>, 4> result;
    result[0].normal    = n4[0];
    result[1].normal    = n4[1];
    result[2].normal    = n4[2];
    result[3].normal    = n4[3];

    result[0].view_dir  = vd4[0];
    result[1].view_dir  = vd4[1];
    result[2].view_dir  = vd4[2];
    result[3].view_dir  = vd4[3];

    result[0].active    = active[0] != 0;
    result[1].active    = active[1] != 0;
    result[2].active    = active[2] != 0;
    result[3].active    = active[3] != 0;

    return result;

}

#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

template <typename L>
struct shade_record<L, simd::float8> : public shade_record_base<L, simd::float8>
{
    simd::mask8 active;
};

#endif


//-------------------------------------------------------------------------------------------------
// Shade record factory
//

namespace detail
{

template <typename Params, typename T>
VSNRAY_FUNC
inline auto make_shade_record(has_textures_tag)
    -> shade_record<typename Params::light_type, vector<3, T>, T>
{
    return shade_record<typename Params::light_type, vector<3, T>, T>();
}

template <typename Params, typename T>
VSNRAY_FUNC
inline auto make_shade_record(has_no_textures_tag)
    -> shade_record<typename Params::light_type, T>
{
    return shade_record<typename Params::light_type, T>();
}

} // detail

template <typename Params, typename T>
VSNRAY_FUNC
inline auto make_shade_record()
    -> decltype( detail::make_shade_record<Params, T>(detail::has_textures<Params>{}) )
{
    return detail::make_shade_record<Params, T>(detail::has_textures<Params>{});
}

} // visionaray

#endif // VSNRAY_SHADE_RECORD_H


