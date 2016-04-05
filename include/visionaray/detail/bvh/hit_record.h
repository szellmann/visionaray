// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_BVH_HIT_RECORD_H
#define VSNRAY_DETAIL_BVH_HIT_RECORD_H 1

#include <type_traits>

#include <visionaray/math/math.h>
#include <visionaray/bvh.h>

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// A special hit record for BVHs that stores additional hit information associated
// with the BVH intersection and that inherits from the hit record for the
// primitive stored by the BVH
//

template <
    typename R,
    typename BVH,
    typename Base,
    typename = typename std::enable_if<is_any_bvh<BVH>::value>::type
    >
struct hit_record_bvh : Base
{
    using bvh_type = BVH;
    using int_type = typename simd::int_type<typename R::scalar_type>::type;

    VSNRAY_FUNC hit_record_bvh() = default;
    VSNRAY_FUNC explicit hit_record_bvh(
            Base const& base,
            int_type    i)
        : Base(base)
        , primitive_list_index(i)
    {
    }

    // Index into the primitive list stored by the bvh
    // Is in general different from primitive::prim_id
    int_type primitive_list_index = 0;
};


//-------------------------------------------------------------------------------------------------
// update_if() overload that dispatches to update_if() for Base in addition
// to store BVH hit information
//

template <typename R, typename BVH, typename Base, typename Cond>
VSNRAY_FUNC
void update_if(
    hit_record_bvh<R, BVH, Base>&       dst,
    hit_record_bvh<R, BVH, Base> const& src,
    Cond const&                         cond
    )
{
    update_if(static_cast<Base&>(dst), static_cast<Base const&>(src), cond);
    dst.primitive_list_index = select( cond, src.primitive_list_index, dst.primitive_list_index );
}


//-------------------------------------------------------------------------------------------------
// simd::unpack()
//

namespace simd
{

template <
    typename FloatT,
    typename BVH,
    typename Base,
    typename = typename std::enable_if<is_simd_vector<FloatT>::value>::type
    >
inline auto unpack(
        hit_record_bvh<basic_ray<FloatT>, BVH, Base> const& hr
        )
    -> std::array<
            hit_record_bvh<ray, BVH, typename decltype(simd::unpack(Base{}))::value_type>,
            num_elements<FloatT>::value
            >
{
    using int_array        = typename aligned_array<typename int_type<FloatT>::type>::type;
    using scalar_base_type = typename decltype(simd::unpack(Base{}))::value_type;

    auto base = simd::unpack(static_cast<Base const&>(hr));

    int_array primitive_list_index;
    store(primitive_list_index, hr.primitive_list_index);

    std::array<
        hit_record_bvh<ray, BVH, scalar_base_type>,
        num_elements<FloatT>::value
        > result;

    for (size_t i = 0; i < num_elements<FloatT>::value; ++i)
    {
        result[i] = hit_record_bvh<ray, BVH, scalar_base_type>(
                scalar_base_type(base[i]),
                primitive_list_index[i]
                );
    }
    return result;
}

} // simd

} // visionaray

#endif // VSNRAY_DETAIL_BVH_HIT_RECORD_H
