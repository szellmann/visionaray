// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <iostream>
#include <ostream>

#include <thrust/device_vector.h>

#include <visionaray/texture/texture.h>
#include <visionaray/aligned_vector.h>

#include <gtest/gtest.h>

using namespace visionaray;


//-------------------------------------------------------------------------------------------------
// host texture typedefs
//

using h_texture_R8          = visionaray::texture<          unorm<8> , NormalizedFloat, 1>;
using h_texture_RG8         = visionaray::texture<vector<2, unorm<8>>, NormalizedFloat, 1>;
using h_texture_RGB8        = visionaray::texture<vector<3, unorm<8>>, NormalizedFloat, 1>;
using h_texture_RGBA8       = visionaray::texture<vector<4, unorm<8>>, NormalizedFloat, 1>;


//-------------------------------------------------------------------------------------------------
// device texture typedefs
//

using d_texture_R8          = cuda_texture<              unorm<8> , NormalizedFloat, 1>;
using d_texture_RG8         = cuda_texture<    vector<2, unorm<8>>, NormalizedFloat, 1>;
using d_texture_RGB8        = cuda_texture<    vector<3, unorm<8>>, NormalizedFloat, 1>;
using d_texture_RGBA8       = cuda_texture<    vector<4, unorm<8>>, NormalizedFloat, 1>;

using d_texture_R32F        = cuda_texture<              float    , NormalizedFloat, 1>;
using d_texture_RG32F       = cuda_texture<    vector<2, float>   , NormalizedFloat, 1>;


//-------------------------------------------------------------------------------------------------
// device texture ref typedefs
//

using d_texture_ref_R8      = cuda_texture_ref<          unorm<8> , NormalizedFloat, 1>;
using d_texture_ref_RG8     = cuda_texture_ref<vector<2, unorm<8>>, NormalizedFloat, 1>;
using d_texture_ref_RGB8    = cuda_texture_ref<vector<3, unorm<8>>, NormalizedFloat, 1>;
using d_texture_ref_RGBA8   = cuda_texture_ref<vector<4, unorm<8>>, NormalizedFloat, 1>;

using d_texture_ref_R32F    = cuda_texture_ref<          float    , NormalizedFloat, 1>;
using d_texture_ref_RG32F   = cuda_texture_ref<vector<2, float  > , NormalizedFloat, 1>;


//-------------------------------------------------------------------------------------------------
// sampler kernel
//

template <
    typename T,
    tex_read_mode ReadMode,
    typename FloatT
    >
__global__ void sample1D(
        cuda_texture_ref<T, ReadMode, 1>    tex,
        FloatT*                             coords,
        T*                                  result,
        size_t                              n
        )
{
    for (size_t i = 0; i < n; ++i)
    {
        result[i] = tex1D(tex, coords[i]);
    }
}


//-------------------------------------------------------------------------------------------------
// Make index based on tex coord and address mode
//

int make_index(float coord, int texsize, tex_address_mode address_mode)
{
    if (address_mode == Clamp)
    {
        auto index = int(coord * texsize);
        return max(0, min(index, texsize - 1));
    }
    else if (address_mode == Wrap)
    {
        // wrap
        auto index = int(coord * texsize);
        while (index < 0)
        {
            index += texsize;
        }
        return index % texsize;
    }
    else if (address_mode == Mirror)
    {
        auto index = int(coord * texsize);
        if (index < 0)
        {
            index += 1;
            index *= -1;
        }
        else if (index >= texsize)
        {
            int tmp = index - texsize;
            index -= tmp * 2;
            index -= 1;
        }
        return index;
    }

    return -1;
}


//-------------------------------------------------------------------------------------------------
// Raw data
//

struct sampler_R8
{
    enum { TexSize   = 8 };
    enum { NumCoords = 22 };

    sampler_R8()
        : d_coords(h_coords, h_coords + NumCoords)
        , d_result(NumCoords)
        , h_texture(TexSize)
    {
        h_texture.set_data( reinterpret_cast<unorm<8> const*>(data) );

        reset();
    }

    void set_address_mode(tex_address_mode address_mode)
    {
        h_texture.set_address_mode(address_mode);

        reset();
    }

    void set_filter_mode(tex_filter_mode filter_mode)
    {
        h_texture.set_filter_mode(filter_mode);

        reset();
    }

    void reset()
    {
        d_texture = d_texture_R8(h_texture);
        d_texture_ref = d_texture_ref_R8(d_texture);
    }


    //-------------------------------------------------------------------------
    // Sample into member arrays
    //

    void sample()
    {
        sample1D<<<1, 1>>>(
                d_texture_ref,
                thrust::raw_pointer_cast(d_coords.data()),
                thrust::raw_pointer_cast(d_result.data()),
                d_coords.size()
                );

        h_result = thrust::host_vector<unorm<8>>(d_result);
    }


    //-------------------------------------------------------------------------
    // Sample w/o using the member arrays
    //

    void sample(
            thrust::host_vector<float> const&   coords,
            thrust::host_vector<unorm<8>>&      result
            )
    {
        thrust::device_vector<float>    c(coords);
        thrust::device_vector<unorm<8>> r(result.size());

        sample1D<<<1, 1>>>(
                d_texture_ref,
                thrust::raw_pointer_cast(c.data()),
                thrust::raw_pointer_cast(r.data()),
                c.size()
                );

        result = thrust::host_vector<unorm<8>>(r);
    }


    static unsigned char const      data[TexSize];

    static float const              h_coords[NumCoords];
    thrust::device_vector<float>    d_coords;

    thrust::device_vector<unorm<8>> d_result;
    thrust::host_vector<unorm<8>>   h_result;


    h_texture_R8                    h_texture;
    d_texture_R8                    d_texture;
    d_texture_ref_R8                d_texture_ref;
};

unsigned char const sampler_R8::data[] = { 0, 16, 32, 64, 96, 128, 255, 255 };

float const sampler_R8::h_coords[] = {
       -3.0f / sampler_R8::TexSize,                                 // underflow
       -2.0f / sampler_R8::TexSize,
       -1.0f / sampler_R8::TexSize,

        0.0f / sampler_R8::TexSize,                                 // valid [0.0..1.0) coords
        0.5f / sampler_R8::TexSize,
        1.0f / sampler_R8::TexSize,
        1.5f / sampler_R8::TexSize,
        2.0f / sampler_R8::TexSize,
        2.5f / sampler_R8::TexSize,
        3.0f / sampler_R8::TexSize,
        3.5f / sampler_R8::TexSize,
        4.0f / sampler_R8::TexSize,
        4.5f / sampler_R8::TexSize,
        5.0f / sampler_R8::TexSize,
        5.5f / sampler_R8::TexSize,
        6.0f / sampler_R8::TexSize,
        6.5f / sampler_R8::TexSize,
        7.0f / sampler_R8::TexSize,

        8.0f / sampler_R8::TexSize - 1.0f / sampler_R8::TexSize,    // last valid value

        8.0f / sampler_R8::TexSize,                                 // overflow
        9.0f / sampler_R8::TexSize,
       10.0f / sampler_R8::TexSize
        };


//-------------------------------------------------------------------------------------------------
// Print lots of samples for debugging
//

template <typename Sampler>
void generate_samples(Sampler& sampler, bool on_device = true, int num_coords = 1024)
{
    thrust::host_vector<float> coords(num_coords);
    thrust::host_vector<unorm<8>> result(num_coords);

    for (int i = 0; i < num_coords; ++i)
    {
        coords[i] = static_cast<float>(i) / num_coords;
    }

    if (on_device)
    {
        sampler.sample(coords, result);
    }
    else
    {
        for (int i = 0; i < num_coords; ++i)
        {
            result[i] = (float)tex1D(sampler.h_texture, coords[i]);
        }
    }

    for (auto r : result)
    {
        std::cout << static_cast<float>(r) << '\n';
    }
}


//-------------------------------------------------------------------------------------------------
// Test CUDA 1-D textures
// Test that device texture lookups and host texture lookups produce equal results
//

TEST(TextureCU, Tex1DR8NormalizedFloatNearest)
{
    //-------------------------------------------------------------------------
    // R8, normalized float, nearest neighbor
    //

    sampler_R8 sampler;
    sampler.set_filter_mode(Nearest);

    // address mode clamp

    sampler.set_address_mode(Clamp);
    sampler.sample();

    for (int i = 0; i < sampler.NumCoords; ++i)
    {
        auto index = make_index(sampler.h_coords[i], sampler.TexSize, Clamp);

        float expected = sampler.data[index] / 255.0f;
        // host
        EXPECT_FLOAT_EQ(expected, tex1D(sampler.h_texture, sampler.h_coords[i]));
        // device
        EXPECT_FLOAT_EQ(expected, sampler.h_result[i]);
    }


    // address mode wrap

    sampler.set_address_mode(Wrap);
    sampler.sample();

    for (int i = 0; i < sampler.NumCoords; ++i)
    {
        auto index = make_index(sampler.h_coords[i], sampler.TexSize, Wrap);

        float expected = sampler.data[index] / 255.0f;
        // host
        EXPECT_FLOAT_EQ(expected, tex1D(sampler.h_texture, sampler.h_coords[i]));
        // device
        EXPECT_FLOAT_EQ(expected, sampler.h_result[i]);
    }


    // address mode mirror

    sampler.set_address_mode(Mirror);
    sampler.sample();

    for (int i = 0; i < sampler.NumCoords; ++i)
    {
        auto index = make_index(sampler.h_coords[i], sampler.TexSize, Mirror);

        float expected = sampler.data[index] / 255.0f;
        // host
        EXPECT_FLOAT_EQ(expected, tex1D(sampler.h_texture, sampler.h_coords[i]));
        // device
        EXPECT_FLOAT_EQ(expected, sampler.h_result[i]);
    }
}


TEST(TextureCU, Tex1DR8NormalizedFloatLinear)
{
    //-------------------------------------------------------------------------
    // R8, normalized float, linear
    //

    sampler_R8 sampler;
    sampler.set_filter_mode(Linear);

    // address mode clamp

    sampler.set_address_mode(Clamp);
    sampler.sample();

    for (int i = 0; i < sampler.NumCoords; ++i)
    {
        auto index = make_index(sampler.h_coords[i], sampler.TexSize, Clamp);

        float expected = tex1D(sampler.h_texture, sampler.h_coords[i]);

        // check if CPU lookup matches GPU lookup
        EXPECT_FLOAT_EQ(expected, sampler.h_result[i]);
    }


    // address mode wrap

    sampler.set_address_mode(Wrap);
    sampler.sample();

    for (int i = 0; i < sampler.NumCoords; ++i)
    {
        auto index = make_index(sampler.h_coords[i], sampler.TexSize, Wrap);

        float expected = tex1D(sampler.h_texture, sampler.h_coords[i]);

        // check if CPU lookup matches GPU lookup
        EXPECT_FLOAT_EQ(expected, sampler.h_result[i]);
    }


    // address mode mirror

    sampler.set_address_mode(Mirror);
    sampler.sample();

    for (int i = 0; i < sampler.NumCoords; ++i)
    {
        auto index = make_index(sampler.h_coords[i], sampler.TexSize, Mirror);

        float expected = tex1D(sampler.h_texture, sampler.h_coords[i]);

        // check if CPU lookup matches GPU lookup
        EXPECT_FLOAT_EQ(expected, sampler.h_result[i]);
    }
}
