// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cassert>

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <utility>

#ifndef NDEBUG
#include <iostream>
#include <ostream>
#endif

#include "aligned_vector.h"
#include "stack.h"

namespace visionaray
{
namespace detail
{


//-------------------------------------------------------------------------------------------------
// create an empty bounding box (min is positive, max is negative)
//

aabb empty_box()
{
    return aabb( vec3(std::numeric_limits<float>::max()), -vec3(std::numeric_limits<float>::max()) );
}


//-------------------------------------------------------------------------------------------------
// aabb of a single triangle
//

aabb bounds(basic_triangle<3, float> const& tri)
{
    auto v1 = tri.v1;
    // TODO
    auto v2 = tri.v1 + tri.e1;
    auto v3 = tri.v1 + tri.e2;

    return aabb
    (
        vec3
        (
            min(v1.x, min(v2.x, v3.x) ),
            min(v1.y, min(v2.y, v3.y) ),
            min(v1.z, min(v2.z, v3.z) )
        ),
        vec3
        (
            max(v1.x, max(v2.x, v3.x) ),
            max(v1.y, max(v2.y, v3.y) ),
            max(v1.z, max(v2.z, v3.z) )
        )
    );
}


//-------------------------------------------------------------------------------------------------
// aabb of a list of aabbs
//

aabb bounds(std::vector<aabb> const& boxes)
{
    auto result = empty_box();

    for (auto const& b : boxes)
    {
        result.min = min( b.min, min(b.max, result.min) );
        result.max = max( b.min, max(b.max, result.max) );
    }

    return result;
}


//-------------------------------------------------------------------------------------------------
// aabb of points
//

aabb bounds(std::vector<vec3> const& points)
{
    auto result = empty_box();

    for (auto const& p : points)
    {
        result.min = min( p, result.min );
        result.max = max( p, result.max );
    }

    return result;
}

aabb bounds(vec3 const* begin, vec3 const* end)
{
    typedef vec3 const* point_iterator;

    auto result = empty_box();

    for (point_iterator it = begin; it != end; ++it)
    {
        result.min = min( *it, result.min );
        result.max = max( *it, result.max );
    }

    return result;
}


//-------------------------------------------------------------------------------------------------
// get the cartesian coordinate of the widest side of an aabb
//

cartesian_axis<3> widest_side(aabb const& box)
{
    auto s = box.size();
    if (s.x > s.y && s.x > s.z)
    {
        return cartesian_axis<3>::X;
    }
    else if (s.y > s.x && s.y > s.z)
    {
        return cartesian_axis<3>::Y;
    }
    else
    {
        return cartesian_axis<3>::Z;
    }
}


//-------------------------------------------------------------------------------------------------
// surface area of a bounding box
//

float area(aabb const& box)
{
    auto s = box.size();
    return float(2.0) * (s.x * s.y + s.y * s.z + s.z * s.x);
}


//-------------------------------------------------------------------------------------------------
// volume of a bounding box
//

float volume(aabb const& box)
{
    auto s = box.size();
    return s.x * s.y * s.z;
}


//-------------------------------------------------------------------------------------------------
// calc bin id of primitive
//

unsigned bin_id(vec3 const& centroid, cartesian_axis<3> const& axis, float k0, float k1)
{
    return k1 * (centroid[axis] - k0);
}

} // detail

template <typename P>
bvh<P> build(P const* primitives, size_t num_prims)
{

    using namespace detail;

    bvh<P> result(primitives, num_prims);

    struct prim_data
    {
        unsigned idx;
        aabb bbox;
        vec3 centroid;
    };

    aligned_vector<prim_data, 64> data(num_prims);

    // circumvent potentially slow operator[] of STL containers
    auto ptr = &data[0];
    auto pptr = &primitives[0];

    aabb bbox = empty_box();
    for (size_t i = 0; i < num_prims; ++i)
    {
        auto p          = pptr[i];
        ptr[i].idx      = i;
        ptr[i].bbox     = bounds(p);
        ptr[i].centroid = ptr[i].bbox.center();
        bbox            = combine( bbox, ptr[i].bbox );
    }

    // index into bvh's preallocated node list
    unsigned current_idx = 0;

    // support only MAX_UINT primitives per bvh
    assert( (num_prims & std::numeric_limits<unsigned>::max()) == num_prims );

    bvh_node root;
    root.bbox       = bbox;
    root.first_prim = 0;
    root.num_prims  = static_cast<unsigned>(num_prims);
    result.nodes()[current_idx] = root;
    unsigned node_stack_ptr = 1;

    stack<64> st;

    auto node = root;

    for (;;)
    {
        auto cb = node.bbox;
        auto axis = widest_side(cb);
        auto first_prim = node.first_prim;
        auto num_prims  = node.num_prims;
        auto prim_begin = first_prim;
        auto prim_end   = first_prim + num_prims;

        // costs for making a leaf (area(cb) is disregarded during binning and must
        // thus be minded for the primitive costs)
        float leaf_costs = /* constant factor for prim isect * */ num_prims * area(cb);

        static const size_t NumBins = 16;

        std::array<aabb,     NumBins> bin_bounds;
        std::array<unsigned, NumBins> bin_counts;
        for (size_t i = 0; i < NumBins; ++i)
        {
            bin_bounds[i] = empty_box();
            bin_counts[i] = 0;
        }

//        return (num_bins * (1 - std::numeric_limits<float>::epsilon()) * (centroid[axis] - cb.min[axis])) / (cb.max[axis] - cb.min[axis]);
        float k0 = cb.min[axis];
        float k1 = (NumBins * (1 - std::numeric_limits<float>::epsilon())) / (cb.max[axis] - cb.min[axis]);
        for (auto i = prim_begin; i < prim_end; ++i)
        {
            auto const& b = ptr[i].bbox;
//            auto bi = bin_id(NumBins, cb, ptr[i].centroid, axis);//if (current_idx == 11) { std::cerr << b.center() << " " << bi << std::endl; }
            auto bi = bin_id(ptr[i].centroid, axis, k0, k1);
//            assert( bi < NumBins );
            if (bi == NumBins)
            {
//                std::cerr << i << std::endl;
                bi = NumBins - 1;
            }
            if (bi > NumBins) { bi = 0; }
            bin_bounds[bi] = combine(bin_bounds[bi], b);
            bin_counts[bi]++;
        }

        // pass from the left: accumulate bounds and counts
        std::array<aabb,     NumBins> acc_bounds_l;
        std::array<unsigned, NumBins> acc_counts_l;

        acc_bounds_l[0] = bin_bounds[0];
        acc_counts_l[0] = bin_counts[0];

        for (size_t i = 1; i < NumBins; ++i)
        {
            acc_bounds_l[i] = combine(acc_bounds_l[i - 1], bin_bounds[i]);
            acc_counts_l[i] = acc_counts_l[i - 1] + bin_counts[i];
        }

        // pass from the right: no need to store accumulated values

        aabb   acc_bounds_r   = bin_bounds[NumBins - 1];
        unsigned acc_counts_r = bin_counts[NumBins - 1];
        std::array<float, NumBins - 1> costs;

        float min_costs = std::numeric_limits<float>::max();
        aabb best_l = empty_box();
        aabb best_r = empty_box();
        unsigned split_plane = std::numeric_limits<unsigned>::max(); // in [0..NumBins)
        unsigned count_l = 0;

        for (size_t i = NumBins - 2; static_cast<int>(i) >= 0; --i)
        {
            costs[i] = area(acc_bounds_l[i]) * acc_counts_l[i] + area(acc_bounds_r) * acc_counts_r;

            float c = max( 0.0f, min(min_costs, costs[i]) );
            if (c > float(0.0) && c < min_costs)
            {
                min_costs   = c;
                best_l      = acc_bounds_l[i];
                best_r      = acc_bounds_r;
                split_plane = i;
                count_l     = acc_counts_l[i];
            }

            acc_bounds_r = combine(acc_bounds_r, bin_bounds[i]);
            acc_counts_r += bin_counts[i];
        }

        if (min_costs < leaf_costs && num_prims > 4 && volume(best_l) > float(0.0) && volume(best_r) > float(0.0))
        {
            // two new nodes
            bvh_node left;
            left.bbox        = best_l;
            left.first_prim  = first_prim;
            left.num_prims   = count_l;

            bvh_node right;
            right.bbox       = best_r;
            right.first_prim = first_prim + count_l;
            right.num_prims  = num_prims - count_l;

            // partition primitive ids
            std::partition
            (
                 &ptr[prim_begin], &ptr[prim_end],
                 [&](prim_data const& pd)
                 {
//                     return bin_id(NumBins, cb, pd.centroid, axis) <= split_plane;
                    return bin_id(pd.centroid, axis, k0, k1) <= split_plane;
                 }
            );

/*            unsigned cnt_l = 0;
            unsigned cnt_r = 0;
            for (auto i = prim_begin; i != prim_end; ++i)
            {
                auto bi = bin_id(NumBins, cb, cptr[i], axis);
                if (bi <= split_plane) cnt_l++;
                if (bi >  split_plane) cnt_r++;
            }

            if (cnt_l != left.num_prims)
            {std::cerr << current_idx << std::endl;
                std::cerr << "l: " << cnt_l << " " << left.num_prims << std::endl;
            }

            if (cnt_r != right.num_prims)
            {std::cerr << current_idx << std::endl;
                std::cerr << "r: " << cnt_r << " " << right.num_prims << std::endl;
            }*/

            // make current node an inner node
            bvh_node& current_node = result.nodes()[current_idx];
            current_node.first_child = node_stack_ptr;
            current_node.num_prims = 0;

            // store new nodes
            result.nodes()[node_stack_ptr]     = left;
            result.nodes()[node_stack_ptr + 1] = right;//std::cerr << "+++push: " << node_stack_ptr + 1 << ' ';std::cerr << result.nodes()[node_stack_ptr + 1].num_prims << std::endl;

            node = left;
            st.push(node_stack_ptr + 1);

            current_idx = node_stack_ptr;
            node_stack_ptr += 2;
        }
        else
        {
            if (st.empty())
            {
                break;
            }
            else
            {
                current_idx = st.pop();//std::cerr << "---popp: " << current_idx << ' ';std::cerr << result.nodes()[current_idx].num_prims << std::endl;
                node = result.nodes()[current_idx];
            }
        }
    }

// TODO: use templates
#ifdef BVH_WITH_GATHER
    for (size_t i = 0; i < num_prims; ++i)
    {
        result.prim_indices()[i] = ptr[i].idx;
    }
#else
    // TODO!!!!!
    auto tmp = const_cast<P*>(result.primitives());
    for (size_t i = 0; i < num_prims; ++i)
    {
        tmp[i] = primitives[ptr[i].idx];
    }
#endif

    // FIXME: this returns a copy!
    // w/o return value optimization, delete[] in d'tor will be called!
    return result;

}


} // visionaray


