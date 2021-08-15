// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include "../simd/type_traits.h"
#include "../aabb.h"
#include "../config.h"
#include "../rectangle.h"

namespace MATH_NAMESPACE
{

//-------------------------------------------------------------------------------------------------
// Triangle members
//

template <size_t Dim, typename T, typename P>
MATH_FUNC
basic_triangle<Dim, T, P>::basic_triangle(
        vector<Dim, T> const& v1,
        vector<Dim, T> const& e1,
        vector<Dim, T> const& e2
        )
    : v1(v1)
    , e1(e1)
    , e2(e2)
{
}


//-------------------------------------------------------------------------------------------------
// Geometric functions
//

template <size_t Dim, typename T, typename P>
MATH_FUNC
inline T area(basic_triangle<Dim, T, P> const& t)
{
    return T(0.5) * length(cross(t.e1, t.e2));
}

template <size_t Dim, typename T, typename P>
MATH_FUNC
inline basic_aabb<T> get_bounds(basic_triangle<Dim, T, P> const& t)
{
    basic_aabb<T> bounds;

    bounds.invalidate();
    bounds.insert(t.v1);
    bounds.insert(t.v1 + t.e1);
    bounds.insert(t.v1 + t.e2);

    return bounds;
}

template <typename T, typename P>
MATH_FUNC
inline rectangle<min_max_layout<2, T>, T> get_bounds(basic_triangle<2, T, P> const& t)
{
    rectangle<min_max_layout<2, T>, T> bounds;

    bounds.invalidate();
    bounds.insert(t.v1);
    bounds.insert(t.v1 + t.e1);
    bounds.insert(t.v1 + t.e2);

    return bounds;
}

template <size_t Dim, typename T, typename P, typename Generator, typename U = typename Generator::value_type>
MATH_FUNC
inline vector<3, U> sample_surface(basic_triangle<Dim, T, P> const& t, Generator& gen)
{
    U u1 = gen.next();
    U u2 = gen.next();

    vector<3, U> v1(t.v1);
    vector<3, U> v2(t.v1 + t.e1);
    vector<3, U> v3(t.v1 + t.e2);

    return v1 * (U(1.0) - sqrt(u1)) + v2 * sqrt(u1) * (U(1.0) - u2) + v3 * sqrt(u1) * u2;
}

template <size_t Dim, typename T, typename P>
MATH_FUNC
inline array<vector<Dim, T>,3> compute_vertices(basic_triangle<Dim, T, P> const& t)
{
    return {{ t.v1, t.v1 + t.e1, t.v1 + t.e2 }};
}

namespace simd
{

//-------------------------------------------------------------------------------------------------
// SIMD conversions
//

// pack ---------------------------------------------------

template <typename T, typename P, size_t N> // only for 3D triangle!
MATH_FUNC
inline basic_triangle<3, float_from_simd_width_t<N>, int_from_simd_width_t<N>> pack(
        array<basic_triangle<3, T, P>, N> const& tris
        )
{
    using U = float_from_simd_width_t<N>; // TODO: generalize, not just float!
    using I = int_from_simd_width_t<N>;

    basic_triangle<3, U, I> result;

    int* prim_id = reinterpret_cast<int*>(&result.prim_id);
    int* geom_id = reinterpret_cast<int*>(&result.geom_id);

    T* v1x = reinterpret_cast<T*>(&result.v1.x);
    T* v1y = reinterpret_cast<T*>(&result.v1.y);
    T* v1z = reinterpret_cast<T*>(&result.v1.z);

    T* e1x = reinterpret_cast<T*>(&result.e1.x);
    T* e1y = reinterpret_cast<T*>(&result.e1.y);
    T* e1z = reinterpret_cast<T*>(&result.e1.z);

    T* e2x = reinterpret_cast<T*>(&result.e2.x);
    T* e2y = reinterpret_cast<T*>(&result.e2.y);
    T* e2z = reinterpret_cast<T*>(&result.e2.z);

    for (size_t i = 0; i < N; ++i)
    {
        prim_id[i] = tris[i].prim_id;
        geom_id[i] = tris[i].geom_id;

        v1x[i] = tris[i].v1.x;
        v1y[i] = tris[i].v1.y;
        v1z[i] = tris[i].v1.z;

        e1x[i] = tris[i].e1.x;
        e1y[i] = tris[i].e1.y;
        e1z[i] = tris[i].e1.z;

        e2x[i] = tris[i].e2.x;
        e2y[i] = tris[i].e2.y;
        e2z[i] = tris[i].e2.z;
    }

    return result;
}

} // simd
} // MATH_NAMESPACE
