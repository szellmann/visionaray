// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_PARALLEL_ALGORITHM_H
#define VSNRAY_DETAIL_PARALLEL_ALGORITHM_H 1

#include <visionaray/config.h>

#include <algorithm>
#include <functional>
#include <iterator>
#include <type_traits>

#if VSNRAY_HAVE_TBB
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#endif

#include "macros.h"

namespace visionaray
{
namespace paralgo
{

//-------------------------------------------------------------------------------------------------
// counting_sort trivial key
//

namespace detail
{

struct trivial_key
{
    template <
        typename T,
        typename = typename std::enable_if<std::is_integral<T>::value>::type
        >
    T operator()(T val)
    {
        return val;
    }
};

} // detail

#if VSNRAY_HAVE_TBB

//-------------------------------------------------------------------------------------------------
// counting_sort
//
// Sorts items based on integer keys.
//
// [in] FIRST
//      Start of the input sequence.
//
// [in] LAST
//      End of the input sequence.
//
// [out] OUT
//      Start of the output sequence.
//
// [in] KEY
//      Sort key function object.
//
// Complexity: O(n + k)
//

namespace detail
{

template <size_t K, typename InputIt, typename Key>
struct concurrent_histogram
{
    concurrent_histogram(InputIt first, Key key)
        : values_{0}
        , first_(first)
        , key_(key)
    {
    }

    concurrent_histogram(concurrent_histogram& rhs, tbb::split)
        : values_{0}
        , first_(rhs.first_)
        , key_(rhs.key_)
    {
    }

    void operator()(tbb::blocked_range<int> const& r)
    {
        for (int i = r.begin(); i < r.end(); ++i)
        {
            ++values_[key_(first_[i])];
        }
    }

    void join(concurrent_histogram const& rhs)
    {
        for (size_t m = 0; m < K; ++m)
        {
            values_[m] += rhs.values_[m];
        }
    }

    unsigned values_[K];
    InputIt first_;
    Key key_;
};

} // detail

template <
    size_t K,
    typename InputIt,
    typename OutputIt,
    typename Key = detail::trivial_key
    >
void counting_sort(InputIt first, InputIt last, OutputIt out, Key key = Key())
{
    static_assert(
            std::is_integral<decltype(key(*first))>::value,
            "parallel_counting_sort requires integral key type"
            );

    detail::concurrent_histogram<K, InputIt, Key> h(first, key);

    tbb::parallel_reduce(tbb::blocked_range<int>(0, last - first), h);

    auto& cnt = h.values_;

    for (size_t m = 1; m < K; ++m)
    {
        cnt[m] += cnt[m - 1];
    }

    for (auto it = first; it != last; ++it)
    {
        out[--cnt[key(*it)]] = *it;
    }
}

#endif // VSNRAY_HAVE_TBB

} // namespace paralgo
} // namespace visionaray

#endif // VSNRAY_DETAIL_PARALLEL_ALGORITHM_H
