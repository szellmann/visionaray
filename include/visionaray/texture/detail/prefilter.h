// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_TEXTURE_DETAIL_PREFILTER_H
#define VSNRAY_TEXTURE_DETAIL_PREFILTER_H 1

#include <limits>
#include <stdexcept>

#include <visionaray/math/math.h>

#include "texture1d.h"
#include "texture2d.h"
#include "texture3d.h"


namespace visionaray
{
namespace detail
{

//-------------------------------------------------------------------------------------------------
// Prefilter for B-Spline interpolation
// Ported from http://dannyruijters.nl/docs/cudaPrefilter3.pdf
//

static float const Pole = sqrt(3.0f) - 2.0f;

inline float init_causal_coeff(short* c, size_t len, size_t stride)
{

    typedef float float_type;
//  typedef short voxel_type;

    size_t const Horizon = std::min<size_t>(12, len);

    float_type zk(Pole);
    float_type sum = float_type(*c) ;
    for (size_t k = 0; k < Horizon; ++k)
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
static void convert_to_bspline_coeffs(T*, size_t, size_t)
{
    throw std::runtime_error("not implemented yet");
}

static void convert_to_bspline_coeffs(short* c, size_t len, size_t stride)
{

    typedef float float_type;
    typedef short voxel_type;

    static float_type const Lambda = 6.0;

    // causal

    *c = voxel_type( Lambda * init_causal_coeff(c, len, stride) );

    for (size_t k = 1; k < len; ++k)
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


template <typename T, tex_read_mode ReadMode>
static void convert_for_bspline_interpol(texture_ref<T, ReadMode, 1>* tex)
{
    using namespace detail;

    T* ptr = &(*tex)(0);
    convert_to_bspline_coeffs(ptr, tex->size(), 1);
}

template <typename T, tex_read_mode ReadMode>
static void convert_for_bspline_interpol(texture_ref<T, ReadMode, 2>* tex)
{
    using namespace detail;

    // row-wise
    for (size_t row = 0; row < tex->height(); ++row)
    {
        T* ptr = &(*tex)[row * tex->width()]; // TODO: use operator()
        convert_to_bspline_coeffs(ptr, tex->width(), 1);
    }

    // column-wise
    for (size_t col = 0; col < tex->width(); ++col)
    {
        T* ptr = &(*tex)[col]; // TODO: use operator()
        convert_to_bspline_coeffs(ptr, tex->height(), tex->width());
    }
}

template <typename T, tex_read_mode ReadMode>
static void convert_for_bspline_interpol(texture_ref<T, ReadMode, 3>* tex)
{
    using namespace detail;

    short* tmp = tex->prefiltered_data;

    for (size_t z = 0; z < tex->depth(); ++z)
    {
        for (size_t y = 0; y < tex->height(); ++y)
        {
            short* ptr = &tmp[z * tex->width() * tex->height() + y * tex->width()];
            convert_to_bspline_coeffs(ptr, tex->width(), 1);
        }
    }

    for (size_t x = 0; x < tex->width(); ++x)
    {
        for (size_t z = 0; z < tex->depth(); ++z)
        {
            short* ptr = &tmp[z * tex->width() * tex->height() + x];
            convert_to_bspline_coeffs(ptr, tex->height(), tex->width());
        }
    }

    for (size_t y = 0; y < tex->height(); ++y)
    {
        for (size_t x = 0; x < tex->width(); ++x)
        {
            short* ptr = &tmp[y * tex->width() + x];
            convert_to_bspline_coeffs(ptr, tex->depth(), tex->width() * tex->height());
        }
    }
}

} // visionaray

#endif // VSNRAY_TEXTURE_DETAIL_PREFILTER_H
