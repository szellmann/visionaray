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

    vector<3, T> isect_pos;
    vector<3, T> normal;
    vector<3, T> view_dir;
    vector<3, T> light_dir;
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
    C tex_color;
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
    C tex_color;
    simd::mask4 active;
};

namespace simd
{

//-------------------------------------------------------------------------------------------------
// Unpack SSE shade record
//

template <typename L>
std::array<shade_record<L, float>, 4> unpack(shade_record<L, float4> const& sr)
{
    auto isect_pos4 = unpack(sr.isect_pos);
    auto normal4    = unpack(sr.normal);
    auto view_dir4  = unpack(sr.view_dir);
    auto light_dir4 = unpack(sr.light_dir);

    VSNRAY_ALIGN(16) int active[4];
    simd::store(active, sr.active.i);

    std::array<shade_record<L, float>, 4> result;

    for (unsigned i = 0; i < 4; ++i)
    {
        result[i].isect_pos = isect_pos4[i];
        result[i].normal    = normal4[i];
        result[i].view_dir  = view_dir4[i];
        result[i].light_dir = light_dir4[i];
        result[i].light     = sr.light;
        result[i].active    = active[i] != 0;
    }

    return result;
}


//-------------------------------------------------------------------------------------------------
// Unpack SSE shade record with texture color
//

template <typename L>
std::array<shade_record<L, vector<3, float>, float>, 4> unpack(shade_record<L, vector<3, float4>, float4> const& sr)
{
    auto isect_pos4 = unpack(sr.isect_pos);
    auto normal4    = unpack(sr.normal);
    auto view_dir4  = unpack(sr.view_dir);
    auto light_dir4 = unpack(sr.light_dir);
    auto tex_color4 = unpack(sr.tex_color);

    VSNRAY_ALIGN(16) int active[4];
    simd::store(active, sr.active.i);

    std::array<shade_record<L, vector<3, float>, float>, 4> result;

    for (unsigned i = 0; i < 4; ++i)
    {
        result[i].isect_pos = isect_pos4[i];
        result[i].normal    = normal4[i];
        result[i].view_dir  = view_dir4[i];
        result[i].light_dir = light_dir4[i];
        result[i].light     = sr.light;
        result[i].tex_color = tex_color4[i];
        result[i].active    = active[i] != 0;
    }

    return result;
}

} // simd

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
namespace srf
{

struct R2 {};
struct R1 : R2 {};

template <class P, class T> auto test_shade_record_type(R1)
    -> decltype(
        std::declval<P>().textures,
        shade_record<typename P::light_type, vector<3, T>, T>()
       );

template <class P, class T> auto test_shade_record_type(R2)
    -> shade_record<typename P::light_type, T>;

template <class P, class T>
using shade_record_type = decltype(test_shade_record_type<P, T>(R1()));

} // srf
} // detail

template <typename Params, typename T>
VSNRAY_FUNC
inline auto make_shade_record() -> detail::srf::shade_record_type<Params, T>
{
    return {};
}

} // visionaray

#endif // VSNRAY_SHADE_RECORD_H
