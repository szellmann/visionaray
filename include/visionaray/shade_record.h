// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_SHADE_RECORD_H
#define VSNRAY_SHADE_RECORD_H 1

#include <type_traits>
#include <utility>

#include "detail/tags.h"
#include "math/simd/type_traits.h"
#include "math/vector.h"
#include "array.h"

namespace visionaray
{

template <typename L, typename T, typename ...Args>
struct shade_record_base
{
    using scalar_type = T;

    vector<3, T> isect_pos;
    vector<3, T> normal;
    vector<3, T> geometric_normal;
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
    typename T,
    typename = typename std::enable_if<is_simd_vector<T>::value>::type,
    typename = typename std::enable_if<
        std::is_floating_point<element_type_t<T>>::value
        >::type
    >
VSNRAY_FUNC
inline array<shade_record<L, float>, num_elements<T>::value> unpack(shade_record<L, T> const& sr)
{
    auto isect_pos        = unpack(sr.isect_pos);
    auto normal           = unpack(sr.normal);
    auto geometric_normal = unpack(sr.geometric_normal);
    auto view_dir         = unpack(sr.view_dir);
    auto light_dir        = unpack(sr.light_dir);

    array<shade_record<L, float>, num_elements<T>::value> result;

    for (unsigned i = 0; i < num_elements<T>::value; ++i)
    {
        result[i].isect_pos        = isect_pos[i];
        result[i].normal           = normal[i];
        result[i].geometric_normal = geometric_normal[i];
        result[i].view_dir         = view_dir[i];
        result[i].light_dir        = light_dir[i];
        result[i].light            = sr.light;
    }

    return result;
}


//-------------------------------------------------------------------------------------------------
// Unpack SIMD shade record with texture color
//

template <
    typename L,
    typename T,
    typename = typename std::enable_if<is_simd_vector<T>::value>::type,
    typename = typename std::enable_if<
        std::is_floating_point<element_type_t<T>>::value
        >::type
    >
VSNRAY_FUNC
inline array<shade_record<L, vector<3, float>, float>, num_elements<T>::value> unpack(
        shade_record<L, vector<3, T>, T> const& sr
        )
{
    auto isect_pos        = unpack(sr.isect_pos);
    auto normal           = unpack(sr.normal);
    auto geometric_normal = unpack(sr.geometric_normal);
    auto view_dir         = unpack(sr.view_dir);
    auto light_dir        = unpack(sr.light_dir);
    auto tex_color        = unpack(sr.tex_color);

    array<shade_record<L, vector<3, float>, float>, num_elements<T>::value> result;

    for (unsigned i = 0; i < num_elements<T>::value; ++i)
    {
        result[i].isect_pos        = isect_pos[i];
        result[i].normal           = normal[i];
        result[i].geometric_normal = geometric_normal[i];
        result[i].view_dir         = view_dir[i];
        result[i].light_dir        = light_dir[i];
        result[i].light            = sr.light;
        result[i].tex_color        = tex_color[i];
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

template <typename P, typename T>
VSNRAY_FUNC
inline auto test_shade_record_type(R1)
    -> decltype(
        std::declval<P>().textures,
        shade_record<typename P::light_type, vector<3, T>, T>()
       );

template <typename P, typename T>
VSNRAY_FUNC
inline auto test_shade_record_type(R2)
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
