// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <utility>

#include "../simd/type_traits.h"
#include "../axis.h"
#include "../config.h"
#include "../limits.h"


namespace MATH_NAMESPACE
{

//--------------------------------------------------------------------------------------------------
// aabb members
//

template <typename T, size_t Dim>
MATH_FUNC
inline basic_aabb<T, Dim>::basic_aabb(vector<Dim, T> const& min, vector<Dim, T> const& max)
    : min(min)
    , max(max)
{
}

template <typename T, size_t Dim>
template <typename U>
MATH_FUNC
inline basic_aabb<T, Dim>::basic_aabb(basic_aabb<U, Dim> const& rhs)
    : min(rhs.min)
    , max(rhs.max)
{
}

template <typename T, size_t Dim>
template <typename U>
MATH_FUNC
inline basic_aabb<T, Dim>::basic_aabb(vector<Dim, U> const& min, vector<Dim, U> const& max)
    : min(min)
    , max(max)
{
}

template <typename T, size_t Dim>
template <typename U>
MATH_FUNC
inline basic_aabb<T, Dim>& basic_aabb<T, Dim>::operator=(basic_aabb<U, Dim> const& rhs)
{
    min = vector<Dim, T>(rhs.min);
    max = vector<Dim, T>(rhs.max);

    return *this;
}

template <typename T, size_t Dim>
MATH_FUNC
inline vector<Dim, T> basic_aabb<T, Dim>::center() const
{
    return (max + min) * value_type(0.5);
}

template <typename T, size_t Dim>
MATH_FUNC
inline vector<Dim, T> basic_aabb<T, Dim>::size() const
{
    return max - min;
}

template <typename T, size_t Dim>
MATH_FUNC
inline vector<Dim, T> basic_aabb<T, Dim>::safe_size() const
{
    auto s = max - min;

    s.x = MATH_NAMESPACE::max(T(0.0), s.x);
    s.y = MATH_NAMESPACE::max(T(0.0), s.y);
    s.z = MATH_NAMESPACE::max(T(0.0), s.z);

    return s;
}

template <typename T, size_t Dim>
MATH_FUNC
inline void basic_aabb<T, Dim>::invalidate()
{
    min = vec_type(numeric_limits<T>::max());
    max = vec_type(numeric_limits<T>::lowest());
}

template <typename T, size_t Dim>
MATH_FUNC
inline bool basic_aabb<T, Dim>::invalid() const
{
    static_assert(Dim == 3, "Size mismatch");

    return min.x > max.x || min.y > max.y || min.z > max.z;
}

template <typename T, size_t Dim>
MATH_FUNC
inline bool basic_aabb<T, Dim>::valid() const
{
    static_assert(Dim == 3, "Size mismatch");

    return min.x <= max.x && min.y <= max.y && min.z <= max.z;
}

template <typename T, size_t Dim>
MATH_FUNC
inline bool basic_aabb<T, Dim>::empty() const
{
    static_assert(Dim == 3, "Size mismatch");

    return min.x >= max.x || min.y >= max.y || min.z >= max.z;
}

template <typename T, size_t Dim>
MATH_FUNC
inline bool basic_aabb<T, Dim>::contains(vector<Dim, T> const& v) const
{
    static_assert(Dim == 3, "Size mismatch");

    return v.x >= min.x && v.x <= max.x
        && v.y >= min.y && v.y <= max.y
        && v.z >= min.z && v.z <= max.z;
}

template <typename T, size_t Dim>
MATH_FUNC
inline bool basic_aabb<T, Dim>::contains(basic_aabb<T, Dim> const& b) const
{
    return contains(b.min) && contains(b.max);
}

template <typename T, size_t Dim>
MATH_FUNC
inline void basic_aabb<T, Dim>::insert(vec_type const& v)
{
    min = MATH_NAMESPACE::min(min, v);
    max = MATH_NAMESPACE::max(max, v);
}

template <typename T, size_t Dim>
MATH_FUNC
inline void basic_aabb<T, Dim>::insert(basic_aabb const& v)
{
    min = MATH_NAMESPACE::min(min, v.min);
    max = MATH_NAMESPACE::max(max, v.max);
}


//--------------------------------------------------------------------------------------------------
// Comparisons
//

template <typename T, size_t Dim>
MATH_FUNC
inline auto operator==(basic_aabb<T, Dim> const& lhs, basic_aabb<T, Dim> const& rhs)
    -> decltype(lhs.min == rhs.min)
{
    return lhs.min == rhs.min && lhs.max == rhs.max;
}

template <typename T, size_t Dim>
MATH_FUNC
inline auto operator!=(basic_aabb<T, Dim> const& lhs, basic_aabb<T, Dim> const& rhs)
    -> decltype(lhs.min != rhs.min)
{
    return lhs.min != rhs.min || lhs.max != rhs.max;
}


//--------------------------------------------------------------------------------------------------
// Geometric functions
//

template <typename T, size_t Dim>
MATH_FUNC
inline basic_aabb<T, Dim> combine(basic_aabb<T, Dim> const& a, basic_aabb<T, Dim> const& b)
{
    return basic_aabb<T>(min(a.min, b.min), max(a.max, b.max));
}

template <typename T, size_t Dim>
MATH_FUNC
inline basic_aabb<T, Dim> combine(basic_aabb<T, Dim> const& a, vector<Dim, T> const& b)
{
    return basic_aabb<T>(min(a.min, b), max(a.max, b));
}

template <typename T, size_t Dim>
MATH_FUNC
inline basic_aabb<T, Dim> intersect(basic_aabb<T, Dim> const& a, basic_aabb<T, Dim> const& b)
{
    return basic_aabb<T, Dim>(max(a.min, b.min), min(a.max, b.max));
}

template <typename T, size_t Dim>
MATH_FUNC
inline T half_surface_area(basic_aabb<T, Dim> const& box)
{
    static_assert(Dim == 3, "Size mismatch");

    auto s = box.size();
    return s.x * s.y + s.y * s.z + s.z * s.x;
}

template <typename T, size_t Dim>
MATH_FUNC
inline T safe_half_surface_area(basic_aabb<T, Dim> const& box)
{
    static_assert(Dim == 3, "Size mismatch");

    auto s = box.safe_size();
    return s.x * s.y + s.y * s.z + s.z * s.x;
}

template <typename T, size_t Dim>
MATH_FUNC
inline T surface_area(basic_aabb<T, Dim> const& box)
{
    return T(2.0) * half_surface_area(box);
}

template <typename T, size_t Dim>
MATH_FUNC
inline T safe_surface_area(basic_aabb<T, Dim> const& box)
{
    return T(2.0) * safe_half_surface_area(box);
}

template <typename T, size_t Dim>
MATH_FUNC
inline T volume(basic_aabb<T, Dim> const& box)
{
    static_assert(Dim == 3, "Size mismatch");

    auto s = box.size();
    return s.x * s.y * s.z;
}

template <typename T, size_t Dim>
MATH_FUNC
inline T overlap_ratio_union(basic_aabb<T, Dim> const& lhs, basic_aabb<T, Dim> const& rhs)
{
    auto I = intersect(lhs, rhs);

    if (I.empty())
    {
        // bounding boxes do not overlap.
        return T(0.0);
    }

    return volume(I) / volume(combine(lhs, rhs));
}

template <typename T, size_t Dim>
MATH_FUNC
inline T overlap_ratio_min(basic_aabb<T, Dim> const& lhs, basic_aabb<T, Dim> const& rhs)
{
    auto I = intersect(lhs, rhs);

    if (lhs.empty() || rhs.empty())
    {
        // an empty bounding box never overlaps another bounding box
        return T(0.0);
    }

    if (I.empty())
    {
        // bounding boxes do not overlap.
        return T(0.0);
    }

    return volume(I) / min(volume(lhs), volume(rhs));
}

template <typename T, size_t Dim>
MATH_FUNC
inline T overlap_ratio(basic_aabb<T, Dim> const& lhs, basic_aabb<T, Dim> const& rhs)
{
//  return overlap_ratio_union(lhs, rhs);
    return overlap_ratio_min(lhs, rhs);
}

template <typename T, size_t Dim>
inline std::pair<basic_aabb<T, Dim>, basic_aabb<T>> split(basic_aabb<T, Dim> const& box, cartesian_axis<3> axis, T splitpos)
{
    static_assert(Dim == 3, "Size mismatch");

    vector<3, T> min1 = box.min;
    vector<3, T> min2 = box.min;
    vector<3, T> max1 = box.max;
    vector<3, T> max2 = box.max;

    max1[axis] = splitpos;
    min2[axis] = splitpos;

    basic_aabb<T> box1(min1, max1);
    basic_aabb<T> box2(min2, max2);
    return std::make_pair(box1, box2);
}

template <typename T, size_t Dim>
MATH_FUNC
inline array<vector<3, T>, 8> compute_vertices(basic_aabb<T, Dim> const& box)
{
    static_assert(Dim == 3, "Size mismatch");

    vector<3, T> min = box.min;
    vector<3, T> max = box.max;

    return {{
        { max.x, max.y, max.z },
        { min.x, max.y, max.z },
        { min.x, min.y, max.z },
        { max.x, min.y, max.z },
        { min.x, max.y, min.z },
        { max.x, max.y, min.z },
        { max.x, min.y, min.z },
        { min.x, min.y, min.z }
    }};
}


namespace simd
{

//-------------------------------------------------------------------------------------------------
// SIMD conversions
//

// pack ---------------------------------------------------

template <typename T, size_t N> // only for 3D aabb!
MATH_FUNC
inline basic_aabb<float_from_simd_width_t<N>, 3> pack(array<basic_aabb<T, 3>, N> const& boxes)
{
    using U = float_from_simd_width_t<N>; // TODO: generalize, not just float!

    basic_aabb<U, 3> result;

    T* minx = reinterpret_cast<T*>(&result.min.x);
    T* miny = reinterpret_cast<T*>(&result.min.y);
    T* minz = reinterpret_cast<T*>(&result.min.z);
    T* maxx = reinterpret_cast<T*>(&result.max.x);
    T* maxy = reinterpret_cast<T*>(&result.max.y);
    T* maxz = reinterpret_cast<T*>(&result.max.z);

    for (size_t i = 0; i < N; ++i)
    {
        minx[i] = boxes[i].min.x;
        miny[i] = boxes[i].min.y;
        minz[i] = boxes[i].min.z;
        maxx[i] = boxes[i].max.x;
        maxy[i] = boxes[i].max.y;
        maxz[i] = boxes[i].max.z;
    }

    return result;
}

} // simd

} // MATH_NAMESPACE
