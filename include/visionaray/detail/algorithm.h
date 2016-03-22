// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_ALGORITHM_H
#define VSNRAY_DETAIL_ALGORITHM_H 1

#include <algorithm>
#include <iterator>

namespace visionaray
{
namespace algo
{

//--------------------------------------------------------------------------------------------------
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
