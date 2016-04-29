// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_TEXTURE_SAMPLER1D_H
#define VSNRAY_TEXTURE_SAMPLER1D_H 1

#include <cstddef>

#include <type_traits>

#include <visionaray/math/math.h>

#include "sampler_common.h"
#include "texture_common.h"


namespace visionaray
{
namespace detail
{


template <
    typename ReturnT,
    typename InternalT,
    typename TexelT,
    typename FloatT,
    typename SizeT
    >
inline ReturnT nearest(
        ReturnT                                 /* */,
        InternalT                               /* */,
        TexelT const*                           tex,
        FloatT                                  coord,
        SizeT                                   texsize,
        std::array<tex_address_mode, 1> const&  address_mode
        )
{
    coord = map_tex_coord(coord, texsize, address_mode);

    auto lo = convert_to_int(coord * convert_to_float(texsize));
    return point(tex, lo, ReturnT());
}


template <
    typename ReturnT,
    typename InternalT,
    typename TexelT,
    typename FloatT,
    typename SizeT
    >
inline ReturnT linear(
        ReturnT                                 /* */,
        InternalT                               /* */,
        TexelT const*                           tex,
        FloatT                                  coord,
        SizeT                                   texsize,
        std::array<tex_address_mode, 1> const&  address_mode
        )
{
    auto coord1 = map_tex_coord(
            coord - FloatT(0.5) / convert_to_float(texsize),
            texsize,
            address_mode
            );

    auto coord2 = map_tex_coord(
            coord + FloatT(0.5) / convert_to_float(texsize),
            texsize,
            address_mode
            );

    auto lo = convert_to_int(coord1 * convert_to_float(texsize));
    auto hi = convert_to_int(coord2 * convert_to_float(texsize));


    InternalT samples[2] =
    {
        InternalT( point(tex, lo, ReturnT()) ),
        InternalT( point(tex, hi, ReturnT()) )
    };

    auto u = coord1 * convert_to_float(texsize) - convert_to_float(lo);

    return ReturnT(lerp(samples[0], samples[1], u));
}


template <
    typename ReturnT,
    typename InternalT,
    typename TexelT,
    typename FloatT,
    typename SizeT
    >
inline ReturnT cubic2(
        ReturnT                                 /* */,
        InternalT                               /* */,
        TexelT const*                           tex,
        FloatT                                  coord,
        SizeT                                   texsize,
        std::array<tex_address_mode, 1> const&  address_mode
        )
{
    bspline::w0_func<FloatT> w0;
    bspline::w1_func<FloatT> w1;
    bspline::w2_func<FloatT> w2;
    bspline::w3_func<FloatT> w3;

    auto x = coord * convert_to_float(texsize) - FloatT(0.5);
    auto floorx = floor(x);
    auto fracx  = x - floor(x);

    auto tmp0 = ( w1(fracx) ) / ( w0(fracx) + w1(fracx) );
    auto h0   = ( floorx - FloatT(0.5) + tmp0 ) / convert_to_float(texsize);

    auto tmp1 = ( w3(fracx) ) / ( w2(fracx) + w3(fracx) );
    auto h1   = ( floorx + FloatT(1.5) + tmp1 ) / convert_to_float(texsize);

    auto f_0  = InternalT( linear(ReturnT(), InternalT(), tex, h0, texsize, address_mode) );
    auto f_1  = InternalT( linear(ReturnT(), InternalT(), tex, h1, texsize, address_mode) );

    return ReturnT(g0(fracx) * f_0 + g1(fracx) * f_1);
}


template <
    typename ReturnT,
    typename InternalT,
    typename TexelT,
    typename FloatT,
    typename SizeT,
    typename W0,
    typename W1,
    typename W2,
    typename W3
    >
inline ReturnT cubic(
        ReturnT                                 /* */,
        InternalT                               /* */,
        TexelT const*                           tex,
        FloatT                                  coord,
        SizeT                                   texsize,
        std::array<tex_address_mode, 1> const&  address_mode,
        W0                                      w0,
        W1                                      w1,
        W2                                      w2,
        W3                                      w3
        )
{
    auto coord1 = map_tex_coord(
            coord - FloatT(1.5) / convert_to_float(texsize),
            texsize,
            address_mode
            );

    auto coord2 = map_tex_coord(
            coord - FloatT(0.5) / convert_to_float(texsize),
            texsize,
            address_mode
            );

    auto coord3 = map_tex_coord(
            coord + FloatT(0.5) / convert_to_float(texsize),
            texsize,
            address_mode
            );

    auto coord4 = map_tex_coord(
            coord + FloatT(1.5) / convert_to_float(texsize),
            texsize,
            address_mode
            );

    decltype(convert_to_int(FloatT{})) pos[4] =
    {
        convert_to_int(coord1 * convert_to_float(texsize)),
        convert_to_int(coord2 * convert_to_float(texsize)),
        convert_to_int(coord3 * convert_to_float(texsize)),
        convert_to_int(coord4 * convert_to_float(texsize))
    };

    auto u = (coord2 * convert_to_float(texsize)) - convert_to_float(pos[1]);

    auto sample = [&](int i) -> InternalT
    {
        return InternalT( point(tex, pos[i], ReturnT()) );
    };

    return ReturnT(w0(u) * sample(0) + w1(u) * sample(1) + w2(u) * sample(2) + w3(u) * sample(3));
}


//-------------------------------------------------------------------------------------------------
// Dispatch function to choose among filtering algorithms
//

template <
    typename ReturnT,
    typename InternalT,
    typename TexelT,
    typename FloatT,
    typename SizeT
    >
inline ReturnT tex1D_impl_choose_filter(
        ReturnT                                 /* */,
        InternalT                               /* */,
        TexelT const&                           tex,
        FloatT                                  coord,
        SizeT                                   texsize,
        tex_filter_mode                         filter_mode,
        std::array<tex_address_mode, 1> const&  address_mode
        )
{
    switch (filter_mode)
    {

    default:
        // fall-through
    case visionaray::Nearest:
        return nearest(
                ReturnT(),
                InternalT(),
                tex,
                coord,
                texsize,
                address_mode
                );

    case visionaray::Linear:
        return linear(
                ReturnT(),
                InternalT(),
                tex,
                coord,
                texsize,
                address_mode
                );

    case visionaray::BSpline:
        return cubic2(
                ReturnT(),
                InternalT(),
                tex,
                coord,
                texsize,
                address_mode
                );

    case visionaray::CardinalSpline:
        return cubic(
                ReturnT(),
                InternalT(),
                tex,
                coord,
                texsize,
                address_mode,
                cspline::w0_func<FloatT>(),
                cspline::w1_func<FloatT>(),
                cspline::w2_func<FloatT>(),
                cspline::w3_func<FloatT>()
                );

    }

}


//-------------------------------------------------------------------------------------------------
// Dispatch function overloads to deduce texture type and internal texture type
//

// float

template <typename T>
inline T tex1D_impl_expand_types(
        T const*                                tex,
        float                                   coord,
        int                                     texsize,
        tex_filter_mode                         filter_mode,
        std::array<tex_address_mode, 1> const&  address_mode
        )
{
    using return_type   = T;
    using internal_type = float;

    return tex1D_impl_choose_filter(
            return_type(),
            internal_type(),
            tex,
            coord,
            texsize,
            filter_mode,
            address_mode
            );
}

template <size_t Dim, typename T>
inline vector<Dim, T> tex1D_impl_expand_types(
        vector<Dim, T> const*                   tex,
        float                                   coord,
        int                                     texsize,
        tex_filter_mode                         filter_mode,
        std::array<tex_address_mode, 1> const&  address_mode
        )
{
    using return_type   = vector<Dim, T>;
    using internal_type = vector<Dim, float>;

    return tex1D_impl_choose_filter(
            return_type(),
            internal_type(),
            tex,
            coord,
            texsize,
            filter_mode,
            address_mode
            );
}


// double

template <typename T>
inline T tex1D_impl_expand_types(
        T const*                                tex,
        double                                  coord,
        int                                     texsize,
        tex_filter_mode                         filter_mode,
        std::array<tex_address_mode, 1> const&  address_mode
        )
{
    using return_type   = T;
    using internal_type = double;

    return tex1D_impl_choose_filter(
            return_type(),
            internal_type(),
            tex,
            coord,
            texsize,
            filter_mode,
            address_mode
            );
}

template <size_t Dim, typename T>
inline vector<Dim, T> tex1D_impl_expand_types(
        vector<Dim, T> const*                   tex,
        double                                  coord,
        int                                     texsize,
        tex_filter_mode                         filter_mode,
        std::array<tex_address_mode, 1> const&  address_mode
        )
{
    using return_type   = vector<Dim, T>;
    using internal_type = vector<Dim, double>;

    return tex1D_impl_choose_filter(
            return_type(),
            internal_type(),
            tex,
            coord,
            texsize,
            filter_mode,
            address_mode
            );
}


// SIMD: AoS textures

template <
    typename T,
    typename FloatT,
    typename = typename std::enable_if<simd::is_simd_vector<FloatT>::value>::type
    >
inline vector<4, FloatT> tex1D_impl_expand_types(
        vector<4, T> const*                             tex,
        FloatT const&                                   coord,
        typename simd::int_type<FloatT>::type const&    texsize,
        tex_filter_mode                                 filter_mode,
        std::array<tex_address_mode, 1> const&          address_mode
        )
{
    using return_type   = vector<4, FloatT>;
    using internal_type = vector<4, FloatT>;

    return tex1D_impl_choose_filter(
            return_type(),
            internal_type(),
            tex,
            coord,
            texsize,
            filter_mode,
            address_mode
            );
}


// SIMD: SoA textures

inline simd::float4 tex1D_impl_expand_types(
        simd::float4 const*                     tex,
        float                                   coord,
        int                                     texsize,
        tex_filter_mode                         filter_mode,
        std::array<tex_address_mode, 1> const&  address_mode
        )
{
    using return_type   = simd::float4;
    using internal_type = simd::float4;

    return tex1D_impl_choose_filter(
            return_type(),
            internal_type(),
            tex,
            coord,
            texsize,
            filter_mode,
            address_mode
            );
}


//-------------------------------------------------------------------------------------------------
// tex1D() dispatch function
//

template <typename Tex, typename FloatT>
inline auto tex1D(Tex const& tex, FloatT coord)
    -> decltype( tex1D_impl_expand_types(
            tex.data(),
            coord,
            FloatT(),
            tex.get_filter_mode(),
            tex.get_address_mode()
            ) )
{
    static_assert(Tex::dimensions == 1, "Incompatible texture type");

    using I = typename simd::int_type<FloatT>::type;

    I texsize = static_cast<int>(tex.width());

    return tex1D_impl_expand_types(
            tex.data(),
            coord,
            texsize,
            tex.get_filter_mode(),
            tex.get_address_mode()
            );
}

} // detail
} // visionaray

#endif // VSNRAY_TEXTURE_SAMPLER1D_H
