// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <type_traits>

#include <visionaray/math/aabb.h>
#include <visionaray/bvh_traits.h>

namespace visionaray
{

template <
    typename BVH,
    typename = typename std::enable_if<is_any_bvh<BVH>::value>::type,
    typename = typename std::enable_if<!is_any_bvh_inst<BVH>::value>::type
    >
MATH_FUNC
aabb get_bounds(BVH const& bvh)
{
    aabb result;
    result.invalidate();

    if (bvh.num_nodes() > 0)
    {
        result = bvh.node(0).get_bounds();
    }

    return result;
}

// Overload for BVH instances
template <
    typename BVH,
    typename = typename std::enable_if<is_any_bvh<BVH>::value>::type,
    typename = typename std::enable_if<is_any_bvh_inst<BVH>::value>::type,
    typename = void
    >
MATH_FUNC
aabb get_bounds(BVH const& bvh)
{
    auto trans = inverse(bvh.transform_inv());

    aabb bbox = get_bounds(bvh.get_ref());

    aabb result;
    result.invalidate();

    auto vertices = compute_vertices(bbox);

    for (vec3 v : vertices)
    {
        v = (trans * vec4(v, 1.0f)).xyz();
        result.insert(v);
    }

    return result;
}

} // visionaray
