// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_BVH_HIT_RECORD_H
#define VSNRAY_DETAIL_BVH_HIT_RECORD_H 1

#include <type_traits>

#include <visionaray/math/simd/type_traits.h>
#include <visionaray/math/ray.h>
#include <visionaray/array.h>
#include <visionaray/update_if.h>

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// A special hit record for BVHs that stores additional hit information associated
// with the BVH intersection and that inherits from the hit record for the
// primitive stored by the BVH
//

template <typename R, typename Base>
struct hit_record_bvh : Base
{
    using scalar_type = typename R::scalar_type;
    using int_type    = simd::int_type_t<scalar_type>;

    VSNRAY_FUNC hit_record_bvh() = default;
    VSNRAY_FUNC explicit hit_record_bvh(Base const& base, int_type i)
        : Base(base)
        , primitive_list_index(i)
    {
    }

    // Index into the primitive list stored by the bvh
    // Is in general different from primitive::prim_id
    int_type primitive_list_index = int_type(0);
};


//-------------------------------------------------------------------------------------------------
// update_if() overload that dispatches to update_if() for Base in addition
// to store BVH hit information
//

template <typename R, typename Base, typename Cond>
VSNRAY_FUNC
void update_if(
    hit_record_bvh<R, Base>&       dst,
    hit_record_bvh<R, Base> const& src,
    Cond const&                    cond
    )
{
    update_if(static_cast<Base&>(dst), static_cast<Base const&>(src), cond);
    dst.primitive_list_index = select( cond, src.primitive_list_index, dst.primitive_list_index );
}


namespace simd
{

//-------------------------------------------------------------------------------------------------
// simd::pack()
//

template <
    size_t N,
    typename T = simd::float_from_simd_width_t<N>,
    typename Base
    >
VSNRAY_FUNC
inline hit_record_bvh<basic_ray<T>, decltype(simd::pack(array<Base, N>{{}}))> pack(
        array<hit_record_bvh<ray, Base>, N> const& hrs
        )
{
    using I = int_type_t<T>;
    using int_array = aligned_array_t<I>;
    using RT = hit_record_bvh<basic_ray<T>, decltype(pack(array<Base, N>{{}}))>;

    array<Base, N> bases;
    int_array primitive_list_index;

    for (size_t i = 0; i < N; ++i)
    {
        // Slicing (on purpose)!
        bases[i] = Base(hrs[i]);
        primitive_list_index[i] = hrs[i].primitive_list_index;
    }

    return RT( pack(bases), I(primitive_list_index) );
}


//-------------------------------------------------------------------------------------------------
// simd::unpack()
//

template <
    typename FloatT,
    typename Base,
    typename UnpackedBase = decltype(unpack(Base{})),
    typename = typename std::enable_if<is_simd_vector<FloatT>::value>::type
    >
VSNRAY_FUNC
inline auto unpack(
        hit_record_bvh<basic_ray<FloatT>, Base> const& hr
        )
    -> array<
            hit_record_bvh<ray, typename UnpackedBase::value_type>,
            num_elements<FloatT>::value
            >
{
    using int_array        = aligned_array_t<int_type_t<FloatT>>;
    using scalar_base_type = typename UnpackedBase::value_type;

    auto base = simd::unpack(static_cast<Base const&>(hr));

    int_array primitive_list_index;
    store(primitive_list_index, hr.primitive_list_index);

    array<
        hit_record_bvh<ray, scalar_base_type>,
        num_elements<FloatT>::value
        > result;

    for (size_t i = 0; i < num_elements<FloatT>::value; ++i)
    {
        result[i] = hit_record_bvh<ray, scalar_base_type>(
                scalar_base_type(base[i]),
                primitive_list_index[i]
                );
    }
    return result;
}

} // simd

} // visionaray

#endif // VSNRAY_DETAIL_BVH_HIT_RECORD_H
