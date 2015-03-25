// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cassert>
#include <array>
#include <vector>


namespace visionaray
{
namespace detail
{


struct binned_sah_builder
{
    enum { NumBins = 16 };

    struct prim_info
    {
        aabb bounds;    // Primitive bounds
        vec3 centroid;  // Center of primitive bounds
        int index;      // Primitive index

        void set(aabb const& bounds, vec3 const& centroid, int index)
        {
            this->bounds = bounds;
            this->centroid = centroid;
            this->index = index;
        }

        void set(aabb const& bounds, int index)
        {
            set(bounds, bounds.center(), index);
        }

        template <class Triangle>
        void set(Triangle const& t, int index)
        {
            bounds.invalidate();
            bounds.insert(t.v1);
            bounds.insert(t.v1 + t.e1);
            bounds.insert(t.v1 + t.e2);

            centroid = bounds.center();

            this->index = index;
        }
    };

    struct prim_data
    {
        std::vector<prim_info> pinfos;
        aabb prim_bounds;
        aabb cent_bounds;

        prim_data() = default;

        template <class I>
        prim_data(I first, I last)
        {
            init(first, last);
        }

        template <class I>
        void init(I first, I last)
        {
            clear();

            pinfos.resize(last - first);

            for (int i = 0; first != last; ++first, ++i)
            {
                pinfos[i].set(*first, i);

                prim_bounds = combine(prim_bounds, pinfos[i].bounds);
                cent_bounds = combine(cent_bounds, pinfos[i].centroid);
            }
        }

        void clear()
        {
            pinfos.clear();
            prim_bounds.invalidate();
            cent_bounds.invalidate();
        }
    };

    struct bin
    {
        aabb prim_bounds;   // Primitive bounds
        aabb cent_bounds;   // Centroid bounds
        int count;          // Number of primitives in this bin

        void clear()
        {
            prim_bounds.invalidate();
            cent_bounds.invalidate();
            count = 0;
        }

        friend bin merge(bin lhs, bin const& rhs)
        {
            lhs.prim_bounds.insert(rhs.prim_bounds);
            lhs.cent_bounds.insert(rhs.cent_bounds);
            lhs.count += rhs.count;

            return lhs;
        }
    };

    using bin_list = std::array<bin, NumBins>;

    static int project_to_bin_unsafe(float t, float k0, float k1)
    {
        // k0 = cb.min
        // k1 = NumBins / (cb.max - cb.min)

        return static_cast<int>(k1 * (t - k0));
    }

    static int project_to_bin(float t, float k0, float k1)
    {
        auto i = project_to_bin_unsafe(t, k0, k1);
        assert(i >= 0);
        return i < NumBins ? i : NumBins - 1;
    }

    template <class PI> // prim_info iterator
    static void project_objects(bin_list& bins, PI first, PI last, float k0, float k1, int axis)
    {
        for (/**/; first != last; ++first)
        {
            auto& b = bins[project_to_bin(first->centroid[axis], k0, k1)];

            b.prim_bounds.insert(first->bounds);
            b.cent_bounds.insert(first->centroid);
            b.count++;
        }
    }

    static float compute_leaf_cost(aabb const& /*prim_bounds*/, int count)
    {
        const float Ci = 3.0f;

        return Ci * count;
    }

    static float compute_sah
    (
        aabb const&     prim_bounds_left,
        int             count_left,
        aabb const&     prim_bounds_right,
        int             count_right,
        float           hsa_p
    )
    {
        float const Ci = 3.0f;
        float const Ct = 1.0f;

        auto hsa_l = half_surface_area(prim_bounds_left);
        auto hsa_r = half_surface_area(prim_bounds_right);

        return Ct + (hsa_l / hsa_p) * (Ci * count_left) + (hsa_r / hsa_p) * (Ci * count_right);
    }

    struct split_result
    {
        aabb prim_bounds[2];    // Primitive bounds for left/right nodes
        aabb cent_bounds[2];    // Centroid bounds for left/right nodes
        float cost;             // Split costs
        int index = -1;         // Split position (bin index)
        int first;              // First index of left node
        int middle;             // Pivot: last index of left node, first index of right node
        int last;               // Last index of right node
    };

    static void find_object_split
    (
        split_result&   sr,
        bin_list const& bins,
        aabb const&     prim_bounds,
        int             first,
        int             last
    )
    {
        auto hsa_p = half_surface_area(prim_bounds);

        auto best_cost = numeric_limits<float>::max();
        auto best_index = -1;

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

            auto const& l = acc_l[i - 1];
            auto const& r = acc_r[i];
            assert(l.count + r.count == last - first);

            auto cost = compute_sah(l.prim_bounds, l.count, r.prim_bounds, r.count, hsa_p);
            assert(cost > 0);

            if (cost < best_cost)
            {
                best_cost = cost;
                best_index = i - 1;
            }
        }

        assert(0 <= best_index && best_index < NumBins - 1);

        auto const& best_l = acc_l[best_index];
        auto const& best_r = acc_r[best_index + 1];

        auto middle = first + best_l.count;

        sr.prim_bounds[0]   = best_l.prim_bounds;
        sr.prim_bounds[1]   = best_r.prim_bounds;
        sr.cent_bounds[0]   = best_l.cent_bounds;
        sr.cent_bounds[1]   = best_r.cent_bounds;
        sr.cost             = best_cost;
        sr.index            = best_index;
        sr.first            = first;
        sr.middle           = first + best_l.count;
        sr.last             = last;

        assert(sr.middle == last - best_r.count);
    }

    template <class P>
    bool split
    (
        split_result&   sr,
        P&              pinfos,
        aabb const&     prim_bounds,
        aabb const&     cent_bounds,
        int             first,
        int             last,
        int             max_leaf_size
    )
    {
        auto count = last - first;

        if (count <= max_leaf_size)
        {
            return false;
        }

        // Find the split axis (TODO: Test all axes...)

        auto size = cent_bounds.size();
        auto axis = max_index(size);

        if (size[axis] <= 0.0f)
        {
            return false;
        }

        // Project the primitives into the bins

        bin_list bins;

        for (auto& b : bins)
        {
            b.clear();
        }

        auto k0 = cent_bounds.min[axis];
        auto k1 = NumBins / size[axis];

        project_objects(bins, pinfos.begin() + first, pinfos.begin() + last, k0, k1, axis);

        // Compute split index

        find_object_split(sr, bins, prim_bounds, first, last);

        // Check if turning this node into a leaf might be better...

        auto leaf_costs = compute_leaf_cost(prim_bounds, count);

        if (sr.cost > leaf_costs)
        {
            return false;
        }

        // Found a new split point.
        // Sort primitive infos.

        auto pivot = std::partition(pinfos.begin() + sr.first, pinfos.begin() + sr.last,
            [&](prim_info const& x)
            {
                return project_to_bin_unsafe(x.centroid[axis], k0, k1) <= sr.index;
            });

        assert(sr.middle - sr.first == pivot - (pinfos.begin() + sr.first));

        return true;
    }
};


} // detail
} // visionaray
