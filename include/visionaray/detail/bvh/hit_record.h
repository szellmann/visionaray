// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_BVH_HIT_RECORD_H
#define VSNRAY_DETAIL_BVH_HIT_RECORD_H 1

#include <cstddef>
#include <type_traits>

#include <visionaray/math/simd/type_traits.h>
#include <visionaray/math/matrix.h>
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
    using base_type   = Base;

    hit_record_bvh() = default;
    VSNRAY_FUNC explicit hit_record_bvh(Base const& base, int_type i)
        : Base(base)
        , primitive_list_index(i)
    {
    }

    // Index into the primitive list stored by the bvh
    // (this list is usually, but not always, accessed with
    // an indirect index by using BVH::primitive() - this index
    // is for *direct* access!)
    int_type primitive_list_index = int_type(0);
};


//-------------------------------------------------------------------------------------------------
// A special hit record for BVH instances
// Stores the transform matrices necessary to transform points and vectors into object space
//

template <typename R, typename Base>
struct hit_record_bvh_inst : hit_record_bvh<R, Base>
{
    using scalar_type = typename R::scalar_type;
    using int_type    = simd::int_type_t<scalar_type>;
    using base_type   = Base;

    hit_record_bvh_inst() = default;
    VSNRAY_FUNC explicit hit_record_bvh_inst(
            hit_record_bvh<R, Base> const& base,
            int_type i,
            matrix<4, 4, scalar_type> const& trans_inv
            )
        : hit_record_bvh<R, Base>(base)
        , primitive_list_index_inst(i)
        , transform_inv(trans_inv)
    {
    }

    // Index into the primitive list stored by the bvh instance
    // (this list is usually, but not always, accessed with
    // an indirect index by using BVH::primitive() - this index
    // is for *direct* access!)
    int_type primitive_list_index_inst = int_type(0);

    // Inverse transformation matrix
    matrix<4, 4, scalar_type> transform_inv = matrix<4, 4, typename R::scalar_type>::identity();
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

// Overload for bvh inst record
template <typename R, typename Base, typename Cond>
VSNRAY_FUNC
void update_if(
    hit_record_bvh_inst<R, Base>&       dst,
    hit_record_bvh_inst<R, Base> const& src,
    Cond const&                         cond
    )
{
    update_if(static_cast<hit_record_bvh<R, Base>&>(dst), static_cast<hit_record_bvh<R, Base> const&>(src), cond);
    dst.primitive_list_index_inst = select( cond, src.primitive_list_index_inst, dst.primitive_list_index_inst );
    dst.transform_inv = select( cond, src.transform_inv, dst.transform_inv );
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
        hit_record_bvh_inst<basic_ray<FloatT>, Base> const& hr
        )
    -> array<
            hit_record_bvh_inst<ray, typename UnpackedBase::value_type>,
            num_elements<FloatT>::value
            >
{
    using int_array        = aligned_array_t<int_type_t<FloatT>>;
    using scalar_base_type = typename UnpackedBase::value_type;

    auto base = simd::unpack(static_cast<Base const&>(hr));

    int_array primitive_list_index;
    store(primitive_list_index, hr.primitive_list_index);

    array<
        hit_record_bvh_inst<ray, scalar_base_type>,
        num_elements<FloatT>::value
        > result;

    int_array primitive_list_index_inst = {};
    store(primitive_list_index_inst, hr.primitive_list_index_inst);

    auto transform_inv = unpack(hr.transform_inv);

    for (size_t i = 0; i < num_elements<FloatT>::value; ++i)
    {
        result[i] = hit_record_bvh_inst<ray, scalar_base_type>(
                hit_record_bvh<basic_ray<float>, scalar_base_type>(
                        scalar_base_type(base[i]),
                        primitive_list_index[i]
                        ),
                primitive_list_index_inst[i],
                transform_inv[i]
                );
    }

    return result;
}

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
