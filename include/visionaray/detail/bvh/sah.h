// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_BVH_SAH_H
#define VSNRAY_DETAIL_BVH_SAH_H 1

#include <cassert>
#include <array>
#include <vector>

#include <visionaray/math/aabb.h>
#include <visionaray/math/sphere.h>
#include <visionaray/math/triangle.h>


namespace visionaray
{
namespace detail
{

inline void split_edge(aabb& L, aabb& R, vec3 const& v0, vec3 const& v1, float plane, int axis)
{
    auto t0 = v0[axis];
    auto t1 = v1[axis];

    if (t0 <= plane)
    {
        L.insert(v0);
    }

    if (t0 >= plane)
    {
        R.insert(v0);
    }

    if ((t0 < plane && plane < t1) || (t1 < plane && plane < t0))
    {
        auto x = lerp(v0, v1, (plane - t0) / (t1 - t0));

        x[axis] = plane; // Fix numerical inaccuracies...

        L.insert(x);
        R.insert(x);
    }
}

} // detail

template <size_t Dim, typename T, typename P>
void split_primitive(aabb& L, aabb& R, float plane, int axis, basic_triangle<Dim, T, P> const& prim)
{
    auto v0 = prim.v1;
    auto v1 = v0 + prim.e1;
    auto v2 = v0 + prim.e2;

    L.invalidate();
    R.invalidate();

    detail::split_edge(L, R, v0, v1, plane, axis);
    detail::split_edge(L, R, v1, v2, plane, axis);
    detail::split_edge(L, R, v2, v0, plane, axis);
}

template <typename T, typename P>
void split_primitive(aabb& L, aabb& R, float plane, int axis, basic_sphere<T, P> const& prim)
{
    auto cen = prim.center;
    auto rad = prim.radius;

    L.invalidate();
    R.invalidate();

    if (plane < cen[axis] - rad)
    {
        // Sphere lies completely on the right side of the clipping plane

        R = aabb(cen - rad, cen + rad);
    }
    else if (plane > cen[axis] + rad)
    {
        // Sphere lies completely on the left side of the clipping plane

        L = aabb(cen - rad, cen + rad);
    }
    else
    {
        vec3 C; // center
        vec3 S; // size/2

        float del = cen[axis] - plane;

        float da = 0.5f * (rad - del);
        float sa = 0.5f * (rad + del);

        float R1 = sqrt(rad * rad - del * del);
        float R2 = rad;

        if (del < 0)
        {
            std::swap(R1, R2);
        }

        // Construct left bounding box

        C = cen;
        C[axis] -= sa;

        S = vec3(R1);
        S[axis] = da;

        L.insert(C - S);
        L.insert(C + S);

        // Construct right bounding box

        C = cen;
        C[axis] += da;

        S = vec3(R2);
        S[axis] = sa;

        R.insert(C - S);
        R.insert(C + S);
    }
}

template <typename Primitive>
void split_primitive(aabb& L, aabb& R, float plane, int axis, Primitive const& prim)
{
    VSNRAY_UNUSED(L);
    VSNRAY_UNUSED(R);
    VSNRAY_UNUSED(plane);
    VSNRAY_UNUSED(axis);
    VSNRAY_UNUSED(prim);

    static_assert(sizeof(Primitive) == 0, "not implemented");
}

} // namespace visionaray


namespace visionaray
{
namespace detail
{


struct sah_builder
{
    // TODO:
    // Add a bounds class with prim_bounds and cent_bounds members...

    struct prim_ref
    {
        aabb bounds; // Primitive bounds
        int index;   // Primitive index

        template <typename Primitive>
        void assign(Primitive const& prim, int i)
        {
            bounds = get_bounds(prim);

            index = i;
        }
    };

    using prim_refs = aligned_vector<prim_ref>;

    template <typename I>
    static void init(prim_refs& refs, aabb& prim_bounds, aabb& cent_bounds, I first, I last)
    {
        refs.resize(last - first);

        prim_bounds.invalidate();
        cent_bounds.invalidate();

        for (int i = 0; first != last; ++first, ++i)
        {
            refs[i].assign(*first, i);

            prim_bounds.insert(refs[i].bounds);
            cent_bounds.insert(refs[i].bounds.center());
        }
    }

    enum
    {
        NumBins = 16
    };

    struct bin
    {
        // TODO:
        // Remove cent_bounds and compute them while partitioning...

        aabb prim_bounds; // Primitive bounds
        aabb cent_bounds; // Centroid bounds
        int enter;        // Number of primitives starting in this bin
        int leave;        // Number of primitives ending in this bin

        void clear()
        {
            prim_bounds.invalidate();
            cent_bounds.invalidate();
            enter = 0;
            leave = 0;
        }

        friend bin merge(bin lhs, bin const& rhs)
        {
            lhs.prim_bounds.insert(rhs.prim_bounds);
            lhs.cent_bounds.insert(rhs.cent_bounds);
            lhs.enter += rhs.enter;
            lhs.leave += rhs.leave;

            return lhs;
        }
    };

    using bin_list = std::array<bin, NumBins>;

    struct projection
    {
        float k0;
        float k1;
        int axis;

        projection(aabb const& bounds, int axis)
            : k0(bounds.min[axis])
            , k1(NumBins / (bounds.max[axis] - k0))
            , axis(axis)
        {
        }

        // Return the bin index of the projected point v.
        // NOTE: The returned index might be out of range!
        int project_unsafe(vec3 const& v) const
        {
            return static_cast<int>(k1 * (v[axis] - k0));
        }

        // Return the bin index of the projected point v.
        // The returned index can be used as an index into a list of bins.
        int project(vec3 const& v) const
        {
            auto i = project_unsafe(v);

            if (i < 0)
            {
                i = 0;
            }

            if (i > NumBins - 1)
            {
                i = NumBins - 1;
            }

            return i;
        }

        // Returns the left plane of the given bin
        float unproject(int i) const
        {
            return i / k1 + k0; // lerp(bounds.min, bounds.max, i/NumBins)
        }
    };

    struct leaf_info
    {
        aabb prim_bounds; // Primitive bounds
        aabb cent_bounds; // Centroid bounds
        int first;        // Index of first primitive reference in this leaf
    };

    using leaf_infos = std::array<leaf_info, 2>;

    static float compute_leaf_cost(int size)
    {
        return 3.0f * size;
    }

    static float compute_split_cost(
        aabb const& bounds_left, int size_left, aabb const& bounds_right, int size_right, float hsa_parent)
    {
        auto hsa_left = safe_half_surface_area(bounds_left);
        auto hsa_right = safe_half_surface_area(bounds_right);

        return 1.0f + (hsa_left / hsa_parent) * compute_leaf_cost(size_left) +
                      (hsa_right / hsa_parent) * compute_leaf_cost(size_right);
    }

    struct split_result
    {
        // TODO:
        // Remove cent_bounds and compute them while partitioning...

        aabb prim_bounds[2]; // Primitive bounds of left/right leaves (computed while binning)
        aabb cent_bounds[2]; // Centroid bounds of left/right leaves (computed while binning)
        int count[2];        // Number of primitive (references) in left/right leaves
        float cost;          // Split cost
        int index;           // Split index (smallest bin index for the right leaf)
    };

    // Uses the given list of bins to find the best split.
    // Returns the information needed to build the left/right subtrees.
    static split_result find_split(bin_list const& bins, aabb const& bounds)
    {
        auto hsa_parent = safe_half_surface_area(bounds);
        assert(hsa_parent > 0);

        auto best_cost = std::numeric_limits<float>::max();
        int best_index = -1;

        // Sweep from left to right.

        bin_list acc_l;

        acc_l[0] = bins[0];

        for (auto i = 1; i < NumBins; ++i)
        {
            acc_l[i] = merge(acc_l[i - 1], bins[i]);
        }

        // Sweep from right to left.

        bin_list acc_r;

        acc_r[NumBins - 1] = bins[NumBins - 1];

        for (auto i = NumBins - 1; i > 0; --i)
        {
            acc_r[i - 1] = merge(acc_r[i], bins[i - 1]);

            auto const& L = acc_l[i - 1];
            auto const& R = acc_r[i];

            auto cost = compute_split_cost(L.prim_bounds, L.enter, R.prim_bounds, R.leave, hsa_parent);
            assert(cost > 0);

            if (cost < best_cost)
            {
                best_cost = cost;
                best_index = i;
            }
        }

        assert(0 < best_index && best_index <= NumBins - 1);

        auto const& L = acc_l[best_index - 1];
        auto const& R = acc_r[best_index];

        split_result sr;

        sr.prim_bounds[0] = L.prim_bounds;
        sr.prim_bounds[1] = R.prim_bounds;
        sr.cent_bounds[0] = L.cent_bounds;
        sr.cent_bounds[1] = R.cent_bounds;
        sr.count[0] = L.enter;
        sr.count[1] = R.leave;
        sr.cost = best_cost;
        sr.index = best_index;

        return sr;
    }

    //--------------------------------------------------------------------------
    // object partition
    //

    // Projects the given primitive into the bins.
    static void project_object(bin_list& bins, prim_ref const& ref, projection pr)
    {
        auto cen = ref.bounds.center();

        auto& b = bins[pr.project(cen)];

        b.prim_bounds.insert(ref.bounds);
        b.cent_bounds.insert(cen);
        b.enter++;
        b.leave++;
    }

    // Find the best object split.
    static split_result find_object_split(prim_refs& refs, leaf_info const& leaf, projection pr)
    {
        bin_list bins;

        for (auto& b : bins)
        {
            b.clear();
        }

        for (auto I = refs.begin() + leaf.first, E = refs.end(); I != E; ++I)
        {
            project_object(bins, *I, pr);
        }

        return find_split(bins, leaf.prim_bounds);
    }

    // Partition the given list of objects
    static void perform_object_partition(
        leaf_infos& childs, split_result const& sr, prim_refs& refs, leaf_info const& leaf, projection pr)
    {
        childs[0].prim_bounds = sr.prim_bounds[0];
        childs[0].cent_bounds = sr.cent_bounds[0];
        childs[1].prim_bounds = sr.prim_bounds[1];
        childs[1].cent_bounds = sr.cent_bounds[1];

        auto pivot = std::partition(
            refs.begin() + leaf.first,
            refs.end(),
            [&](prim_ref const& x)
            {
                return pr.project_unsafe(x.bounds.center()) < sr.index;
            }
        );

        childs[0].first = leaf.first;
        childs[1].first = static_cast<int>(pivot - refs.begin());
    }

    //--------------------------------------------------------------------------
    // spatial split
    //

    template <typename Data>
    static void split_reference(prim_ref& L, prim_ref& R, prim_ref const& ref, float plane, int axis, Data const& data)
    {
        split_primitive(L.bounds, R.bounds, plane, axis, data[ref.index]);

        // Clip with current bounds
        L.bounds = intersect(L.bounds, ref.bounds);
        R.bounds = intersect(R.bounds, ref.bounds);

        L.index = ref.index;
        R.index = ref.index;
    }

    template <typename Data>
    static void split_object(bin_list& bins, prim_ref const& ref, projection pr, Data const& data)
    {
        // Compute the range of bins this triangle overlaps
        auto imin = pr.project(ref.bounds.min);
        auto imax = pr.project(ref.bounds.max);

        // Update all the bins this triangle overlaps

        // This is used to clip the left (and right) bounds of a triangle to the
        // current triangle and the current bin bounds.
        auto clip = ref.bounds;

        for (int i = imin; i < imax; ++i)
        {
            auto plane = pr.unproject(i + 1);

            aabb L, R;

            // Split triangle into left and right bounds
            split_primitive(L, R, plane, pr.axis, data[ref.index]);

            // Clip the left triangle to the current triangle bounds and the current bin bounds
            L = intersect(L, clip);

            // Update the clip bounds
            clip = intersect(R, clip);

            bins[i].prim_bounds.insert(L);
            bins[i].cent_bounds.insert(L.center());
        }

        bins[imax].prim_bounds.insert(clip);
        bins[imax].cent_bounds.insert(clip.center());

        // Update counters
        bins[imin].enter++;
        bins[imax].leave++;
    }

    template <typename Data>
    static split_result
    find_spatial_split(prim_refs const& refs, leaf_info const& leaf, projection pr, Data const& data)
    {
        bin_list bins;

        for (auto& b : bins)
        {
            b.clear();
        }

        for (auto I = refs.begin() + leaf.first, E = refs.end(); I != E; ++I)
        {
            split_object(bins, *I, pr, data);
        }

        return find_split(bins, leaf.prim_bounds);
    }

    template <typename Data>
    static void perform_spatial_split(
            leaf_infos&         childs,
            split_result const& sr,
            prim_refs&          refs,
            leaf_info const&    leaf,
            projection          pr,
            Data const&         data
            )
    {
        auto plane = pr.unproject(sr.index);

        auto pivot = leaf.first;
        auto i = leaf.first;
        auto last = static_cast<int>(refs.size());

        childs[0].prim_bounds.invalidate();
        childs[0].cent_bounds.invalidate();
        childs[1].prim_bounds.invalidate();
        childs[1].cent_bounds.invalidate();

        while (i != last)
        {
            auto pmin = refs[i].bounds.min[pr.axis];
            auto pmax = refs[i].bounds.max[pr.axis];

            if (pmax <= plane)
            {
                // Triangle lies completely to the left of the splitting plane.
                // Swap current reference with current pivot to move it to the correct place.

                childs[0].prim_bounds.insert(refs[i].bounds);
                childs[0].cent_bounds.insert(refs[i].bounds.center());

                // xxxxxxxyyyyyyyx.......
                //        ^      ^
                //        p      i

                if (i != pivot)
                {
                    std::swap(refs[pivot], refs[i]);
                }

                ++pivot;
                ++i;

                // xxxxxxxxyyyyyyy.......
                //         ^      ^
                //         p      i
            }
            else if (pmin >= plane)
            {
                // Triamgle lies completely to the right of the splitting plane.
                // Reference is already at the correct place.

                childs[1].prim_bounds.insert(refs[i].bounds);
                childs[1].cent_bounds.insert(refs[i].bounds.center());

                // xxxxxxxyyyyyyyy.......
                //        ^      ^
                //        p      i

                ++i;

                // xxxxxxxyyyyyyyy.......
                //        ^       ^
                //        p       i
            }
            else
            {
                // Triangle intersects the splitting plane.

                prim_ref L, R;

                split_reference(L, R, refs[i], plane, pr.axis, data);

                // TODO:
                // Reference unsplitting.

                childs[0].prim_bounds.insert(L.bounds);
                childs[0].cent_bounds.insert(L.bounds.center());
                childs[1].prim_bounds.insert(R.bounds);
                childs[1].cent_bounds.insert(R.bounds.center());

                // xxxxxxxyyyyyyyS.......   (S -> x,y)
                //        ^      ^
                //        p      i

                refs[i] = R;

                // xxxxxxxyyyyyyyy.......
                //        ^      ^
                //        p      i

                refs.push_back(L);

                // xxxxxxxyyyyyyyy.......x
                //        ^      ^
                //        p      i

                std::swap(refs[pivot], refs.back());

                ++pivot;
                ++i;

                // xxxxxxxxyyyyyyyy......y
                //         ^       ^
                //         p       i
            }
        }

        childs[0].first = leaf.first;
        childs[1].first = pivot;
    }

    //--------------------------------------------------------------------------
    // split
    //

    // TODO:
    // Remove refs and factor out the object partition code...

    // List of primitives references (will be modified during build)
    prim_refs refs;
    // Surface area threshold for spatial splits
    float sa_threshold = 1.0e+38f;
    // Alpha (relative threshold)
    float alpha = 1.0e-5f;
    // Whether to use spatial splits
    bool use_spatial_splits = false;

    void set_alpha(float value)
    {
        alpha = value;
    }

    void enable_spatial_splits(bool enable)
    {
        use_spatial_splits = enable;
    }

    template <typename I>
    leaf_info init(I first, I last)
    {
        aabb prim_bounds;
        aabb cent_bounds;

        init(refs, prim_bounds, cent_bounds, first, last);

        sa_threshold = alpha * safe_surface_area(prim_bounds);

        return { prim_bounds, cent_bounds, 0 };
    }

    // Inserts primitive indices into INDICES and removes them from the current list.
    template <typename Indices>
    int insert_indices(Indices& indices, leaf_info const& leaf)
    {
        int leaf_size = static_cast<int>(refs.size() - leaf.first);

        // Insert indices
        for (int i = leaf.first; i != (int)refs.size(); ++i)
        {
            indices.push_back(refs[i].index);
        }

        // Erase the no longer used primitive references
        refs.resize(leaf.first);

        return leaf_size;
    }

    // Return true if the leaf should be split into two new leaves. In this case
    // sr.leaves contains the information of the left/right leaves and the
    // method returns true. If the leaf should not be split, returns false.
    template <typename Data>
    bool split(leaf_infos& childs, leaf_info const& leaf, Data const& data, int max_leaf_size)
    {
        // FIXME:
        // Create a leaf if max_depth is reached...
        // Or check this in build_tree?

        auto leaf_size = static_cast<int>(refs.size() - leaf.first);

        if (leaf_size <= max_leaf_size)
        {
            return false;
        }

        // Find the split axis (TODO: Test all axes...)

        // Using centroid bounds for object partitioning...
        auto size = leaf.cent_bounds.size();
        auto axis = max_index(size);

        if (size[axis] <= 0.0f)
        {
            return false;
        }

        // Object split --------------------------------------------------------

        projection pr(leaf.cent_bounds, static_cast<int>(axis));

        auto sr = find_object_split(refs, leaf, pr);

        // Spatial split -------------------------------------------------------

        bool do_spatial_split = false;

        if (use_spatial_splits)
        {
            auto sa = safe_surface_area(intersect(sr.prim_bounds[0], sr.prim_bounds[1]));

            if (sa > sa_threshold)
            {
                // Using primitive bounds for spatial splits...
                size = leaf.prim_bounds.size();
                axis = max_index(size);

                if (size[axis] <= 0.0f)
                {
                    return false;
                }

                projection pr2(leaf.prim_bounds, static_cast<int>(axis));

                auto sr2 = find_spatial_split(refs, leaf, pr2, data);

                if (sr2.cost < sr.cost /* && (sr2.count[0] + sr2.count[1] < 1.5 * leaf_size) */)
                {
                    do_spatial_split = true;
                    pr = pr2;
                    sr = sr2;
                }
            }
        }

        // Check if turning this node into a leaf might be better

        auto leaf_costs = compute_leaf_cost(leaf_size);

        if (sr.cost > leaf_costs)
        {
            return false;
        }

        // Found a new split point.
        // Sort primitive references.

        if (do_spatial_split)
        {
            perform_spatial_split(childs, sr, refs, leaf, pr, data);
        }
        else
        {
            perform_object_partition(childs, sr, refs, leaf, pr);
        }

        return true;
    }
};

} // detail
} // visionaray

#endif // VSNRAY_DETAIL_BVH_SAH_H
