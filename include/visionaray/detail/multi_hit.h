// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_MULTI_HIT_H
#define VSNRAY_DETAIL_MULTI_HIT_H 1

#include <array>
#include <cstddef>
#include <type_traits>

#include "tags.h"

namespace visionaray
{
namespace detail
{

//-------------------------------------------------------------------------------------------------
// traversal_result
//
// Trait class to determine the result type of traversal functions.
// In the general case, this is simply a hit_record.
// In the case of multi_hit, the traversal result is an array of hit records.
//

template <typename HR, traversal_type Traversal>
struct traversal_result
{
    using type = HR;
};

template <typename HR>
struct traversal_result<HR, MultiHit>
{
    using type = std::array<HR, 16/*TODO*/>;
};


//-------------------------------------------------------------------------------------------------
// is_multi_hit_record
//
// Trait class to check if a hit record template parameter is a multi-hit record.
// Assumes that the given parameter is any type of hit record!
//

template <typename HR>
struct is_multi_hit_record
{
    enum { value = false };
};

template <typename HR, size_t N>
struct is_multi_hit_record<std::array<HR, N>>
{
    enum { value = true };
};

} // detail


//-------------------------------------------------------------------------------------------------
// update_if() for multi-hit traversal
//

template <
    typename HR1,
    typename HR2,
    typename std::enable_if< detail::is_multi_hit_record<HR1>::value>::type* = nullptr,
    typename std::enable_if<!detail::is_multi_hit_record<HR2>::value>::type* = nullptr,
    typename Cond
    >
VSNRAY_FUNC
void update_if(HR1& dst, HR2 const& src, Cond const& cond)
{
    VSNRAY_UNUSED(cond);

    algo::insert_sorted(src, std::begin(dst), std::end(dst), is_closer_t());
}

// TODO: find a way to enable_if this function w/o having to
// disable_if<MultiHit> the most general update_if overload
template <typename HR, size_t N, typename Cond>
VSNRAY_FUNC
void update_if(std::array<HR, N>& dst, std::array<HR, N> const& src, Cond const& cond)
{
    VSNRAY_UNUSED(cond);

    if (!dst[0].hit)
    {
        // Optimize for the case that no valid hit was found before
        dst = src;
    }
    else
    {
        for (auto const& hr : src)
        {
            if (!hr.hit)
            {
                break;
            }

            algo::insert_sorted(hr, std::begin(dst), std::end(dst), is_closer_t());
        }
    }
}


//-------------------------------------------------------------------------------------------------
// is_closer() for multi-hit traversal
//
// Test if a single-hit record is closer than any result in the multi-hit reference
//

template <
    typename HR1,
    typename HR2,
    typename std::enable_if<!detail::is_multi_hit_record<HR1>::value>::type* = nullptr,
    typename std::enable_if< detail::is_multi_hit_record<HR2>::value>::type* = nullptr,
    typename T
    >
VSNRAY_FUNC
auto is_closer(HR1 const& query, HR2 const& reference, T max_t)
    -> typename simd::mask_type<T>::type
{
    VSNRAY_UNUSED(query, reference, max_t);

    using RT = typename simd::mask_type<T>::type;

    RT result(false);

    for (size_t i = 0; i < reference.size(); ++i)
    {
        result |= is_closer(query, reference[i], max_t);

        if (all(result))
        {
            break;
        }
    }

    return result;
}

// TODO: rename, this is not is_closer!!
template <
    typename HR1,
    typename HR2,
    typename std::enable_if<detail::is_multi_hit_record<HR1>::value>::type* = nullptr,
    typename std::enable_if<detail::is_multi_hit_record<HR2>::value>::type* = nullptr,
    typename T
    >
VSNRAY_FUNC
auto is_closer(HR1 const& query, HR2 const& reference, T max_t)
    -> typename simd::mask_type<T>::type
{
    VSNRAY_UNUSED(query, reference, max_t);

    using RT = typename simd::mask_type<T>::type;

    return RT(true);
}

} // visionaray

#endif // VSNRAY_DETAIL_MULTI_HIT_H
