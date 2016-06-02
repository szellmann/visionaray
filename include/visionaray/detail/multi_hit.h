// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_MULTI_HIT_H
#define VSNRAY_DETAIL_MULTI_HIT_H 1

#include <array>
#include <cstddef>
#include <type_traits>

#include <visionaray/math/math.h>

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

template <typename HR, traversal_type Traversal, size_t MaxHits>
struct traversal_result
{
    using type = HR;
};

template <typename HR, size_t MaxHits>
struct traversal_result<HR, MultiHit, MaxHits>
{
    using type = std::array<HR, MaxHits>;
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

namespace algo
{

//-------------------------------------------------------------------------------------------------
// SIMD version of insert_sorted()
//
// Checks either for is_simd_vector<T>::value, or T contains a type scalar_type so that
// is_simd_vector<scalar_type>::value holds true
//
// TODO: consolidate w/ scalar version of insert_sorted() / move to a better place
// TODO: fix/consolidate overloading with SFINAE
//

template <
    template <typename, typename...> class HR,
    typename S,
    typename ...Args,
    typename RandIt,
    typename std::enable_if<simd::is_simd_vector<S>::value>::type* = nullptr,
    typename Cond
    >
VSNRAY_FUNC
void insert_sorted(HR<basic_ray<S>, Args...> const& item, RandIt first, RandIt last, Cond cond)
{
    using I = typename simd::int_type<S>::type;
    using M = typename simd::mask_type<S>::type;
    using int_array = typename simd::aligned_array<I>::type;

    int i = 0;
    int length = last - first;

    I pos = length;
    M active = true;

    while (i < length)
    {
        auto insert = cond(item, first[i]) && active;
        pos = select(insert, I(i), pos);
        active = select(insert, M(false), active);

        if (!any(active))
        {
            break;
        }

        ++i;
    }

    i = any(!active) ? length - 1 : 0;

    while (i >= 0)
    {
        M must_shift = I(i) > pos;
        M must_insert = I(i) == pos;

        if (all(must_shift))
        {
            first[i] = first[i - 1];
        }
        else if (any(must_shift))
        {
            int_array mask;
            simd::store(mask, must_shift.i);

            auto dst = unpack(first[i]);
            auto src = unpack(first[i - 1]);

            decltype(dst) arr;
            for (size_t j = 0; j < arr.size(); ++j)
            {
                arr[j] = mask[j] ? src[j] : dst[j];
            }
            first[i] = simd::pack(arr);
        }

        if (all(must_insert))
        {
            first[i] = item;
        }
        else if (any(must_insert))
        {
            int_array mask;
            simd::store(mask, must_insert.i);

            auto dst = unpack(first[i]);
            auto src = unpack(item);

            decltype(dst) arr;
            for (size_t j = 0; j < arr.size(); ++j)
            {
                arr[j] = mask[j] ? src[j] : dst[j];
            }
            first[i] = simd::pack(arr);
        }

        if (!any(must_shift) && !any(must_insert))
        {
            break;
        }

        --i;
    }
}

} // algo


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
// disable_if<MultiHit> the more general update_if overload
template <typename HR, size_t N, typename Cond>
VSNRAY_FUNC
void update_if(std::array<HR, N>& dst, std::array<HR, N> const& src, Cond const& cond)
{
    VSNRAY_UNUSED(cond);

    if (!any(dst[0].hit))
    {
        // Optimize for the case that no valid hit was found before
        dst = src;
    }
    else
    {
        for (auto const& hr : src)
        {
            if (!any(hr.hit))
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
