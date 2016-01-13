// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <algorithm>
#include <cstddef>

namespace visionaray
{
namespace algo
{
namespace detail
{

enum neutralize_result { LeftBlock, RightBlock, BothBlocks };

template <typename T, typename UnaryPredicate, size_t BlockSize>
neutralize_result neutralize(T* left, T* right, UnaryPredicate pred)
{
    size_t i = 0;
    size_t j = 0;

    do
    {
        for (i = 0; i < BlockSize; ++i)
        {
            if (pred(left[i]))
            {
                break;
            }
        }

        for (j = 0; j < BlockSize; ++j)
        {
            if (!pred(right[j])) // ??
            {
                break;
            }
        }

        if (i == BlockSize || j == BlockSize)
        {
            break;
        }

        swap(left[i], right[j]);
        ++i;
        ++j;

    } while (i < BlockSize && j < BlockSize);

    if (i == BlockSize && j == BlockSize)
    {
        return BothBlocks;
    }

    if (i == BlockSize)
    {
        return LeftBlock;
    }

    return RightBlock;
}

} // detail
} // algo
} // visionaray
