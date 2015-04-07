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

namespace simd
{

template <typename L>
std::array<shade_record<L, float>, 4> unpack(shade_record<L, float4> const& sr)
{

    auto n4     = unpack(sr.normal);
    auto vd4    = unpack(sr.view_dir);
    VSNRAY_ALIGN(16) int active[4];
    simd::store(active, sr.active.i);

    std::array<shade_record<L, float>, 4> result;
    result[0].normal    = n4[0];
    result[1].normal    = n4[1];
    result[2].normal    = n4[2];
    result[3].normal    = n4[3];

    result[0].view_dir  = vd4[0];
    result[1].view_dir  = vd4[1];
    result[2].view_dir  = vd4[2];
    result[3].view_dir  = vd4[3];

    result[0].light     = sr.light;
    result[1].light     = sr.light;
    result[2].light     = sr.light;
    result[3].light     = sr.light;

    result[0].active    = active[0] != 0;
    result[1].active    = active[1] != 0;
    result[2].active    = active[2] != 0;
    result[3].active    = active[3] != 0;

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
