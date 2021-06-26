// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_TEXTURE_DETAIL_PREFILTER_H
#define VSNRAY_TEXTURE_DETAIL_PREFILTER_H 1

#include <stdexcept>

#include <visionaray/math/detail/math.h>

#include "texture_common.h"


namespace visionaray
{
namespace detail
{

//-------------------------------------------------------------------------------------------------
// Prefilter for B-Spline interpolation
// Ported from http://dannyruijters.nl/docs/cudaPrefilter3.pdf
//

static float const Pole = sqrt(3.0f) - 2.0f;

inline float init_causal_coeff(short* c, unsigned len, unsigned stride)
{

    typedef float float_type;
//  typedef short voxel_type;

    unsigned const Horizon = min<unsigned>(12, len);

    float_type zk(Pole);
    float_type sum = float_type(*c) ;
    for (unsigned k = 0; k < Horizon; ++k)
    {
        sum += zk * float_type(*c);
        zk *= float_type(Pole);
        c += stride;
    }

    return sum;
}

inline float init_anticausal_coeff(short* c)
{
    typedef float float_type;
    return (Pole / (Pole - float_type(1.0))) * float_type(*c);
}

template <typename T>
inline void convert_to_bspline_coeffs(T*, unsigned, unsigned)
{
    throw std::runtime_error("not implemented yet");
}

inline void convert_to_bspline_coeffs(short* c, unsigned len, unsigned stride)
{

    typedef float float_type;
    typedef short voxel_type;

    static float_type const Lambda = 6.0;

    // causal

    *c = voxel_type( Lambda * init_causal_coeff(c, len, stride) );

    for (unsigned k = 1; k < len; ++k)
    {
        c += stride;
        *c = voxel_type( Lambda * float_type(*c) + Pole * *(c - stride) );
    }

    // anticausal

    *c = voxel_type( init_anticausal_coeff(c) );

    for (ptrdiff_t k = len - 2; 0 <= k; --k)
    {
        c -= stride;
        *c = voxel_type( Pole * (*(c + stride) - float_type(*c)) );
    }

}

} // detail


template <typename T>
inline void convert_for_bspline_interpol(texture_ref<T, 1> const* tex)
{
    using namespace detail;

    T const* ptr = &(*tex)(0);
    convert_to_bspline_coeffs(ptr, tex->size(), 1);
}

template <typename T>
inline void convert_for_bspline_interpol(texture_ref<T, 2> const* tex)
{
    using namespace detail;

    // row-wise
    for (unsigned row = 0; row < tex->height(); ++row)
    {
        T* ptr = &(*tex)[row * tex->width()]; // TODO: use operator()
        convert_to_bspline_coeffs(ptr, tex->width(), 1);
    }

    // column-wise
    for (unsigned col = 0; col < tex->width(); ++col)
    {
        T* ptr = &(*tex)[col]; // TODO: use operator()
        convert_to_bspline_coeffs(ptr, tex->height(), tex->width());
    }
}

template <typename T>
inline void convert_for_bspline_interpol(texture_ref<T, 3> const* tex)
{
    using namespace detail;

    short* tmp = tex->prefiltered_data;

    for (unsigned z = 0; z < tex->depth(); ++z)
    {
        for (unsigned y = 0; y < tex->height(); ++y)
        {
            short* ptr = &tmp[z * tex->width() * tex->height() + y * tex->width()];
            convert_to_bspline_coeffs(ptr, tex->width(), 1);
        }
    }

    for (unsigned x = 0; x < tex->width(); ++x)
    {
        for (unsigned z = 0; z < tex->depth(); ++z)
        {
            short* ptr = &tmp[z * tex->width() * tex->height() + x];
            convert_to_bspline_coeffs(ptr, tex->height(), tex->width());
        }
    }

    for (unsigned y = 0; y < tex->height(); ++y)
    {
        for (unsigned x = 0; x < tex->width(); ++x)
        {
            short* ptr = &tmp[y * tex->width() + x];
            convert_to_bspline_coeffs(ptr, tex->depth(), tex->width() * tex->height());
        }
    }
}

} // visionaray

#endif // VSNRAY_TEXTURE_DETAIL_PREFILTER_H
