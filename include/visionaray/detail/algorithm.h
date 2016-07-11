// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_ALGORITHM_H
#define VSNRAY_DETAIL_ALGORITHM_H 1

#include <algorithm>
#include <iterator>

#include "macros.h"

namespace visionaray
{
namespace algo
{

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

    it = pos ? last - 1 : last;
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
