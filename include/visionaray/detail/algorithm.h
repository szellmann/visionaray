// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_ALGORITHM_H
#define VSNRAY_DETAIL_ALGORITHM_H 1

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
namespace algo
{

//-------------------------------------------------------------------------------------------------
// copy
//
// Copies the range [first,...,last) to another range beginning at d_first.
//
// Custom implementation to run in CUDA kernels, resorts to std::copy otherwise.
//

template <typename T, typename U>
VSNRAY_FUNC
U copy(T first, T last, U d_first)
{
#ifdef __CUDA_ARCH__
    while (first != last)
    {
        *d_first++ = *first++;
    }
    return d_first;
#else
    return std::copy(first, last, d_first);
#endif
}


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
            "counting_sort requires integral key type"
            );

    unsigned cnt[K] = { 0 };

    for (auto it = first; it != last; ++it)
    {
        ++cnt[key(*it)];
    }

    for (size_t m = 1; m < K; ++m)
    {
        cnt[m] += cnt[m - 1];
    }

    for (auto it = first; it != last; ++it)
    {
        out[cnt[key(*it)] - 1] = *it;
        --cnt[key(*it)];
    }
}

#if VSNRAY_HAVE_TBB

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
void parallel_counting_sort(InputIt first, InputIt last, OutputIt out, Key key = Key())
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
        out[cnt[key(*it)] - 1] = *it;
        cnt[key(*it)] -= 1;
    }
}

#endif // VSNRAY_HAVE_TBB


//-------------------------------------------------------------------------------------------------
// insert_sorted
//
// Inserts an element into the sorted sequence [first,...,last) so that the
// sequence is sorted afterwards.
//
// It is the user's responsibility to ensure that the sequence is sorted
// according to the condition argument COND provided to this function!
//
// Parameters:
//
// [in] ITEM
//      The element to insert.
//
// [in,out] FIRST
//      Start of the sorted sequence.
//
// [in,out] LAST
//      End of the sorted sequence.
//
// [in] COND
//      Conditional function that, given to elements, returns the 'smaller'
//      one in terms of the order of the sorted sequence.
//
// Complexity: O(n)
//

template <typename T, typename RandIt, typename Cond>
VSNRAY_FUNC
void insert_sorted(T const& item, RandIt first, RandIt last, Cond cond)
{
    RandIt it = first;
    RandIt pos = last;

    while (it < last)
    {
        if (cond(item, *it))
        {
            pos = it;
            break;
        }
        ++it;
    }

    it = (pos != last) ? last - 1 : last;
    while (it > pos)
    {
        *it = *(it - 1);
        --it;
    }

    if (pos != last)
    {
        *pos = item;
    }
}


//-------------------------------------------------------------------------------------------------
// reorder
//
// Reorders the elements in [data, data + count) according to the indices stored
// in [indices, indices + count).
// The list of indices must form a permutation of [0,1,2,...,count).
//
// Effectively this algorithms sorts the indices while shuffling the elements
// in data accordingly.
//
// Parameters:
//
// [in,out] INDICES
//      On entry, contains a permutation of [0,...,count).
//      On exit, contains the identity permutation [0,1,2,...,count).
//
// [in,out] DATA
//      On entry, contains the list of elements to be sorted according to the
//      list of indices.
//      On exit, contains [data[I[0]], data[I[1]], ...), where I denotes the
//      list of indices on entry.
//
// [in] COUNT
//      The number of indices and data elements.
//      Must be >= 0.
//

template <typename RanIt1, typename RanIt2, typename Int>
void reorder_n(RanIt1 indices, RanIt2 data, Int count)
{
    for (Int i = 0; i < count; ++i)
    {
        auto inext = indices[i];
//      assert(inext >= 0);
//      assert(inext < count);

        if (i == inext)
            continue;

        auto temp = std::move(data[i]);
        auto j = i;
//      auto inextnext = indices[inext];

        for (;;)
        {
//          assert(inextnext != j && "cycle detected");

            indices[j] = j;
            if (inext == i)
            {
                data[j] = std::move(temp);
                break;
            }
            else
            {
                data[j] = std::move(data[inext]);

                j = inext;

                inext = indices[j];
//              assert(inext >= 0);
//              assert(inext < count);

//              inextnext = indices[inext];
//              assert(inextnext >= 0);
//              assert(inextnext < count);
            }
        }
    }
}

template <typename RanIt1, typename RanIt2>
void reorder(RanIt1 indices_first, RanIt1 indices_last, RanIt2 data)
{
    auto count = std::distance(indices_first, indices_last);

    reorder_n(indices_first, data, count);
}

} // namespace algo
} // namespace visionaray

#endif // VSNRAY_DETAIL_ALGORITHM_H
