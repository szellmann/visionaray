// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <array>
#include <limits>

#include "stack.h"
#include "traverse.h"

namespace visionaray
{

template <typename T, typename B>
VSNRAY_FUNC
inline hit_record<basic_ray<T>, primitive<unsigned>> intersect
(
    basic_ray<T> const& ray,
    B const& b
)
{

    using namespace detail;

    hit_record<basic_ray<T>, primitive<unsigned>> result;
    result.hit = T(0.0);
//    result.t = T(std::numeric_limits<double>::max());
    result.t = T(3.402823466e+38f);
    result.prim_id = T(0.0);

    stack<32> st;
    /*static*/ unsigned const Sentinel = unsigned(-1);
    st.push(Sentinel);

    unsigned addr = 0;
    auto node = b.nodes()[addr];

    auto inv_dir = T(1.0) / ray.dir;

    for (;;)
    {
        if (is_leaf(node))
        {
            auto begin = node.first_prim;
            auto end   = node.first_prim + node.num_prims;

            for (auto i = begin; i != end; ++i)
            {
                auto prim = b.primitive(i);

                auto hr  = intersect(ray, prim);
                auto closer = hr.hit & ( hr.t >= T(0.0) && hr.t < result.t );
                result.hit |= closer;
                result.t = select( closer, hr.t, result.t );
                result.prim_type = select( closer, hr.prim_type, result.prim_type );
                result.prim_id   = select( closer, hr.prim_id, result.prim_id );
                result.geom_id   = select( closer, hr.geom_id, result.geom_id );
                result.u         = select( closer, hr.u, result.u );
                result.v         = select( closer, hr.v, result.v );
            }

            addr = st.pop();
        }
        else
        {
            bvh_node const* children = b.nodes() + node.first_child;

            auto hr1 = intersect(ray, children[0].bbox, inv_dir);
            auto hr2 = intersect(ray, children[1].bbox, inv_dir);

            auto b1 = hr1.tnear < hr1.tfar && hr1.tnear < result.t && hr1.tfar >= T(0.0);
            auto b2 = hr2.tnear < hr2.tfar && hr2.tnear < result.t && hr2.tfar >= T(0.0);

            if (any(b1) && any(b2))
            {
                unsigned near_addr = all( hr1.tnear < hr2.tnear ) ? 0 : 1;
                st.push(node.first_child + (!near_addr));
                addr = node.first_child + near_addr;
            }
            else if (any(b1))
            {
                addr = node.first_child;
            }
            else if (any(b2))
            {
                addr = node.first_child + 1;
            }
            else
            {
                addr = st.pop();
            }
        }

        if (addr != Sentinel)
        {
            node = b.nodes()[addr];
        }
        else
        {
            break;
        }

    }

    return result;

}

} // visionaray


