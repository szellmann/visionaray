// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <type_traits>

#include <visionaray/math/aabb.h>

namespace visionaray
{

template <
    typename BVH,
    typename = typename std::enable_if<is_any_bvh<BVH>::value>::type
    >
MATH_FUNC
aabb get_bounds(BVH const& bvh)
{
    aabb result;
    result.invalidate();

    if (bvh.num_nodes() > 0)
    {
        result = bvh.node(0).bbox;
    }

    return result;
}

} // visionaray
