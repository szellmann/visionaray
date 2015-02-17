// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_TEXTURE_SAMPLER2D_H
#define VSNRAY_TEXTURE_SAMPLER2D_H

namespace visionaray
{
namespace detail
{


template <typename T>
inline T index(T x, T y, vector<2, T> texsize)
{
    return y * texsize[0] + x;
}


template <typename ReturnT, typename FloatT, typename TexelT>
inline ReturnT nearest(TexelT const* tex, vector<2, FloatT> coord, vector<2, FloatT> texsize)
{

    using visionaray::clamp;

    typedef vector<2, FloatT> float2;

    float2 lo
    (
        floor(coord.x * texsize.x),
        floor(coord.y * texsize.y)
    );

    lo[0] = clamp(lo[0], FloatT(0.0f), texsize[0] - 1);
    lo[1] = clamp(lo[1], FloatT(0.0f), texsize[1] - 1);

    FloatT idx = index(lo[0], lo[1], texsize);
    return point(tex, idx);

}


template <typename ReturnT, typename FloatT, typename TexelT>
inline ReturnT linear(TexelT const* tex, vector<2, FloatT> coord, vector<2, FloatT> texsize)
{

    typedef vector<2, FloatT> float2;

    float2 texcoordf( coord * texsize - FloatT(0.5) );

    texcoordf[0] = clamp( texcoordf[0], FloatT(0.0), texsize[0] - 1 );
    texcoordf[1] = clamp( texcoordf[1], FloatT(0.0), texsize[1] - 1 );

    float2 lo( floor(texcoordf[0]), floor(texcoordf[1]) );
    float2 hi( ceil(texcoordf[0]),  ceil(texcoordf[1]) );


    vector<3, FloatT> samples[4] = // TODO: this must somehow be consolidated with ReturnT
    {
        vector<3, FloatT>( point(tex, index( lo.x, lo.y, texsize )) ),
        vector<3, FloatT>( point(tex, index( hi.x, lo.y, texsize )) ),
        vector<3, FloatT>( point(tex, index( lo.x, hi.y, texsize )) ),
        vector<3, FloatT>( point(tex, index( hi.x, hi.y, texsize )) )
    };


    float2 uv = texcoordf - lo;

    auto p1 = lerp(samples[0], samples[1], uv[0]);
    auto p2 = lerp(samples[2], samples[3], uv[0]);

    return ReturnT(lerp(p1, p2, uv[1]));

}


template <typename ReturnT, typename FloatT, typename TexelT>
inline ReturnT tex2D(texture_ref<TexelT, ElementType, 2> const& tex, vector<2, FloatT> coord)
{

    vector<2, FloatT> texsize( tex.width(), tex.height() );
    return nearest<ReturnT>( tex.data(), coord, texsize );

}


} // detail
} // visionaray


#endif // VSNRAY_TEXTURE_SAMPLER2D_H
