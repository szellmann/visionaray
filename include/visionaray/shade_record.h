// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_SHADE_RECORD_H
#define VSNRAY_SHADE_RECORD_H 1

#include <array>
#include <type_traits>

#include "detail/tags.h"
#include "math/math.h"

namespace visionaray
{

template <typename L, typename T, typename ...Args>
struct shade_record_base
{
    using scalar_type = T;
    using mask_type = typename simd::mask_type<T>::type;

    mask_type active;
    vector<3, T> isect_pos;
    vector<3, T> normal;
    vector<3, T> view_dir;
    vector<3, T> light_dir;
    L light;
};

template <typename L, typename T, typename ...Args>
struct shade_record : public shade_record_base<L, T, Args...>
{
};

template <typename L, typename C, typename T, typename ...Args>
struct shade_record<L, C, T, Args...> : public shade_record_base<L, T, Args...>
{
    C tex_color;
};

namespace simd
{

//-------------------------------------------------------------------------------------------------
// Unpack SIMD shade record
//

template <
    typename L,
    typename FloatT,
    typename = typename std::enable_if<is_simd_vector<FloatT>::value>::type
    >
inline std::array<shade_record<L, float>, num_elements<FloatT>::value> unpack(
        shade_record<L, FloatT> const& sr
        )
{
    using int_array = typename aligned_array<typename int_type<FloatT>::type>::type;

    auto isect_pos  = unpack(sr.isect_pos);
    auto normal     = unpack(sr.normal);
    auto view_dir   = unpack(sr.view_dir);
    auto light_dir  = unpack(sr.light_dir);

    int_array active;
    store(active, sr.active.i);

    std::array<shade_record<L, float>, num_elements<FloatT>::value> result;

    for (unsigned i = 0; i < num_elements<FloatT>::value; ++i)
    {
        result[i].isect_pos = isect_pos[i];
        result[i].normal    = normal[i];
        result[i].view_dir  = view_dir[i];
        result[i].light_dir = light_dir[i];
        result[i].light     = sr.light;
        result[i].active    = active[i] != 0;
    }

    return result;
}


//-------------------------------------------------------------------------------------------------
// Unpack SIMD shade record with texture color
//

template <
    typename L,
    typename FloatT,
    typename = typename std::enable_if<is_simd_vector<FloatT>::value>::type
    >
inline std::array<shade_record<L, vector<3, float>, float>, num_elements<FloatT>::value> unpack(
        shade_record<L, vector<3, FloatT>, FloatT> const& sr
        )
{
    using int_array = typename aligned_array<typename int_type<FloatT>::type>::type;

    auto isect_pos  = unpack(sr.isect_pos);
    auto normal     = unpack(sr.normal);
    auto view_dir   = unpack(sr.view_dir);
    auto light_dir  = unpack(sr.light_dir);
    auto tex_color  = unpack(sr.tex_color);

    int_array active;
    store(active, sr.active.i);

    std::array<shade_record<L, vector<3, float>, float>, num_elements<FloatT>::value> result;

    for (unsigned i = 0; i < num_elements<FloatT>::value; ++i)
    {
        result[i].isect_pos = isect_pos[i];
        result[i].normal    = normal[i];
        result[i].view_dir  = view_dir[i];
        result[i].light_dir = light_dir[i];
        result[i].light     = sr.light;
        result[i].tex_color = tex_color[i];
        result[i].active    = active[i] != 0;
    }

    return result;
}

} // simd


//-------------------------------------------------------------------------------------------------
// Shade record factory
//

namespace detail
{
namespace srf
{

struct R2 {};
struct R1 : R2 {};

template <typename P, typename T> auto test_shade_record_type(R1)
    -> decltype(
        std::declval<P>().textures,
        shade_record<typename P::light_type, vector<3, T>, T>()
       );

template <typename P, typename T> auto test_shade_record_type(R2)
    -> shade_record<typename P::light_type, T>;

template <typename P, typename T>
using shade_record_type
    = decltype( test_shade_record_type<typename std::decay<P>::type,
                                       typename std::decay<T>::type>(R1()) );

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
