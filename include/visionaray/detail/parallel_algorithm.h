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
// Sorts items based on integer keys in [0..k).
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
// [in,out] COUNTS
//      Modifiable counts sequence.
//
// [in] KEY
//      Sort key function object.
//
// Complexity: TODO: parallel complexity
//

namespace detail
{

template <typename T>
struct test_resizable
{
    template <typename U, U>
    struct test_ {};

    template <typename U>
    static std::true_type test(test_<void (U::*)(size_t), &U::resize>*);

    template <typename U>
    static std::false_type test(...);

    using type = decltype( test<typename std::decay<T>::type>(nullptr) );
};

struct init_values
{
    template <
        typename Cont
        >
    typename std::enable_if<std::is_same<typename test_resizable<Cont>::type, std::true_type>::value>::type*
    operator()(Cont& cont, size_t k)
    {
        cont.resize(k);
        std::fill(std::begin(cont), std::end(cont), 0);
        return nullptr;
    }

    template <
        typename Cont
        >
    typename std::enable_if<std::is_same<typename test_resizable<Cont>::type, std::false_type>::value>::type*
    operator()(Cont& cont, size_t)
    {
        std::fill(std::begin(cont), std::end(cont), 0);
        return nullptr;
    }
};

template <typename InputIt, typename Values, typename Key>
struct concurrent_histogram
{
    concurrent_histogram(InputIt first, size_t k, Key key)
        : first_(first)
        , key_(key)
    {
        init_values()(values_, k);
    }

    concurrent_histogram(concurrent_histogram& rhs, tbb::split)
        : first_(rhs.first_)
        , key_(rhs.key_)
    {
        init_values()(values_, rhs.values_.size());
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
        for (size_t m = 0; m < values_.size(); ++m)
        {
            values_[m] += rhs.values_[m];
        }
    }

    InputIt first_;
    Values values_;
    Key key_;
};

} // detail

template <
    typename InputIt,
    typename OutputIt,
    typename Counts,
    typename Key = detail::trivial_key
    >
void counting_sort(InputIt first, InputIt last, OutputIt out, Counts& counts, Key key = Key())
{
    static_assert(
            std::is_integral<decltype(key(*first))>::value,
            "parallel_counting_sort requires integral key type"
            );

    detail::concurrent_histogram<InputIt, Counts, Key> h(first, counts.size(), key);

    tbb::parallel_reduce(tbb::blocked_range<int>(0, last - first), h);

    auto& cnt = h.values_;

    for (size_t m = 1; m < cnt.size(); ++m)
    {
        cnt[m] += cnt[m - 1];
    }

    std::copy(cnt.begin(), cnt.end(), counts.begin());

    for (auto it = first; it != last; ++it)
    {
        out[--cnt[key(*it)]] = *it;
    }
}

#endif // VSNRAY_HAVE_TBB

} // namespace paralgo
} // namespace visionaray

#endif // VSNRAY_DETAIL_PARALLEL_ALGORITHM_H
