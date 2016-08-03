// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cuda_runtime_api.h>

#include <visionaray/cuda/cast.h>
#include <visionaray/math/unorm.h>
#include <visionaray/math/vector.h>

#include <gtest/gtest.h>

using namespace visionaray;


//-------------------------------------------------------------------------------------------------
// Test casts between CUDA and Visionaray vector types (e.g. float3 <-> vector<3, float>)
//

TEST(CastCU, CudaToVisionaray)
{
    // vec2 -----------------------------------------------

    char2 c2 = make_char2(-96, 127);
    auto cc2 = cuda::cast<vector<2, char>>(c2);
    EXPECT_EQ(c2.x, cc2.x);
    EXPECT_EQ(c2.y, cc2.y);

    uchar2 uc2 = make_uchar2(96, 255);
    auto uuc2 = cuda::cast<vector<2, unsigned char>>(uc2);
    EXPECT_EQ(uc2.x, uuc2.x);
    EXPECT_EQ(uc2.y, uuc2.y);

    short2 s2 = make_short2(-4711, 32767);
    auto ss2 = cuda::cast<vector<2, short>>(s2);
    EXPECT_EQ(s2.x, ss2.x);
    EXPECT_EQ(s2.y, ss2.y);

    ushort2 us2 = make_ushort2(4711, 65535);
    auto uus2 = cuda::cast<vector<2, unsigned short>>(us2);
    EXPECT_EQ(us2.x, uus2.x);
    EXPECT_EQ(us2.y, uus2.y);

    int2 i2 = make_int2(-1024, 2147483647);
    auto ii2 = cuda::cast<vector<2, int>>(i2);
    EXPECT_EQ(i2.x, ii2.x);
    EXPECT_EQ(i2.y, ii2.y);

    uint2 ui2 = make_uint2(1024, 4294967295);
    auto uui2 = cuda::cast<vector<2, unsigned int>>(ui2);
    EXPECT_EQ(ui2.x, uui2.x);
    EXPECT_EQ(ui2.y, uui2.y);

    float2 f2 = make_float2(0.0f, 4711.0f);
    auto ff2 = cuda::cast<vector<2, float>>(f2);
    EXPECT_FLOAT_EQ(f2.x, ff2.x);
    EXPECT_FLOAT_EQ(f2.y, ff2.y);

    // vec3 -----------------------------------------------

    char3 c3 = make_char3(-96, 0, 127);
    auto cc3 = cuda::cast<vector<3, char>>(c3);
    EXPECT_EQ(c3.x, cc3.x);
    EXPECT_EQ(c3.y, cc3.y);
    EXPECT_EQ(c3.z, cc3.z);

    uchar3 uc3 = make_uchar3(0, 96, 255);
    auto uuc3 = cuda::cast<vector<3, unsigned char>>(uc3);
    EXPECT_EQ(uc3.x, uuc3.x);
    EXPECT_EQ(uc3.y, uuc3.y);
    EXPECT_EQ(uc3.z, uuc3.z);

    short3 s3 = make_short3(-4711, 0, 32767);
    auto ss3 = cuda::cast<vector<3, short>>(s3);
    EXPECT_EQ(s3.x, ss3.x);
    EXPECT_EQ(s3.y, ss3.y);
    EXPECT_EQ(s3.z, ss3.z);

    ushort3 us3 = make_ushort3(0, 4711, 65535);
    auto uus3 = cuda::cast<vector<3, unsigned short>>(us3);
    EXPECT_EQ(us3.x, uus3.x);
    EXPECT_EQ(us3.y, uus3.y);
    EXPECT_EQ(us3.z, uus3.z);

    int3 i3 = make_int3(-1024, 0, 2147483647);
    auto ii3 = cuda::cast<vector<3, int>>(i3);
    EXPECT_EQ(i3.x, ii3.x);
    EXPECT_EQ(i3.y, ii3.y);
    EXPECT_EQ(i3.z, ii3.z);

    uint3 ui3 = make_uint3(0, 1024, 4294967295);
    auto uui3 = cuda::cast<vector<3, unsigned int>>(ui3);
    EXPECT_EQ(ui3.x, uui3.x);
    EXPECT_EQ(ui3.y, uui3.y);
    EXPECT_EQ(ui3.z, uui3.z);

    float3 f3 = make_float3(0.0f, 4711.0f, 65535.0f);
    auto ff3 = cuda::cast<vector<3, float>>(f3);
    EXPECT_FLOAT_EQ(f3.x, ff3.x);
    EXPECT_FLOAT_EQ(f3.y, ff3.y);
    EXPECT_FLOAT_EQ(f3.z, ff3.z);

    // vec4 -----------------------------------------------

    char4 c4 = make_char4(-128, -96, 0, 127);
    auto cc4 = cuda::cast<vector<4, char>>(c4);
    EXPECT_EQ(c4.x, cc4.x);
    EXPECT_EQ(c4.y, cc4.y);
    EXPECT_EQ(c4.z, cc4.z);
    EXPECT_EQ(c4.w, cc4.w);

    uchar4 uc4 = make_uchar4(0, 32, 96, 255);
    auto uuc4 = cuda::cast<vector<4, unsigned char>>(uc4);
    EXPECT_EQ(uc4.x, uuc4.x);
    EXPECT_EQ(uc4.y, uuc4.y);
    EXPECT_EQ(uc4.z, uuc4.z);
    EXPECT_EQ(uc4.w, uuc4.w);

    short4 s4 = make_short4(-32768, -4711, 0, 32767);
    auto ss4 = cuda::cast<vector<4, short>>(s4);
    EXPECT_EQ(s4.x, ss4.x);
    EXPECT_EQ(s4.y, ss4.y);
    EXPECT_EQ(s4.z, ss4.z);
    EXPECT_EQ(s4.w, ss4.w);

    ushort4 us4 = make_ushort4(0, 1024, 4711, 65535);
    auto uus4 = cuda::cast<vector<4, unsigned short>>(us4);
    EXPECT_EQ(us4.x, uus4.x);
    EXPECT_EQ(us4.y, uus4.y);
    EXPECT_EQ(us4.z, uus4.z);
    EXPECT_EQ(us4.w, uus4.w);

    int4 i4 = make_int4(-2147483648, -1024, 0, 2147483647);
    auto ii4 = cuda::cast<vector<4, int>>(i4);
    EXPECT_EQ(i4.x, ii4.x);
    EXPECT_EQ(i4.y, ii4.y);
    EXPECT_EQ(i4.z, ii4.z);
    EXPECT_EQ(i4.w, ii4.w);

    uint4 ui4 = make_uint4(0, 1024, 4711, 4294967295);
    auto uui4 = cuda::cast<vector<4, unsigned int>>(ui4);
    EXPECT_EQ(ui4.x, uui4.x);
    EXPECT_EQ(ui4.y, uui4.y);
    EXPECT_EQ(ui4.z, uui4.z);
    EXPECT_EQ(ui4.w, uui4.w);

    float4 f4 = make_float4(0.0f, 4711.0f, 65535.0f, 4294967295.0f);
    auto ff4 = cuda::cast<vector<4, float>>(f4);
    EXPECT_FLOAT_EQ(f4.x, ff4.x);
    EXPECT_FLOAT_EQ(f4.y, ff4.y);
    EXPECT_FLOAT_EQ(f4.z, ff4.z);
    EXPECT_FLOAT_EQ(f4.w, ff4.w);
}

TEST(CastCU, VisionarayToCuda)
{
    // vec2 -----------------------------------------------

    vector<2, char> c2(-96, 127);
    char2 cc2 = cuda::cast<char2>(c2);
    EXPECT_EQ(c2.x, cc2.x);
    EXPECT_EQ(c2.y, cc2.y);

    vector<2, unsigned char> uc2(96, 255);
    uchar2 uuc2 = cuda::cast<uchar2>(uc2);
    EXPECT_EQ(uc2.x, uuc2.x);
    EXPECT_EQ(uc2.y, uuc2.y);

    vector<2, short> s2(-4711, 32767);
    short2 ss2 = cuda::cast<short2>(s2);
    EXPECT_EQ(s2.x, ss2.x);
    EXPECT_EQ(s2.y, ss2.y);

    vector<2, unsigned short> us2(4711, 65535);
    ushort2 uus2 = cuda::cast<ushort2>(us2);
    EXPECT_EQ(us2.x, uus2.x);
    EXPECT_EQ(us2.y, uus2.y);

    vector<2, int> i2(-1024, 2147483647);
    int2 ii2 = cuda::cast<int2>(i2);
    EXPECT_EQ(i2.x, ii2.x);
    EXPECT_EQ(i2.y, ii2.y);

    vector<2, unsigned int> ui2(1024, 4294967295);
    uint2 uui2 = cuda::cast<uint2>(ui2);
    EXPECT_EQ(ui2.x, uui2.x);
    EXPECT_EQ(ui2.y, uui2.y);

    vector<2, float> f2(0.0f, 4711.0f);
    float2 ff2 = cuda::cast<float2>(f2);
    EXPECT_FLOAT_EQ(f2.x, ff2.x);
    EXPECT_FLOAT_EQ(f2.y, ff2.y);

    // vec3 -----------------------------------------------

    vector<3, char> c3(-96, 0, 127);
    char3 cc3 = cuda::cast<char3>(c3);
    EXPECT_EQ(c3.x, cc3.x);
    EXPECT_EQ(c3.y, cc3.y);
    EXPECT_EQ(c3.z, cc3.z);

    vector<3, unsigned char> uc3(0, 96, 255);
    uchar3 uuc3 = cuda::cast<uchar3>(uc3);
    EXPECT_EQ(uc3.x, uuc3.x);
    EXPECT_EQ(uc3.y, uuc3.y);
    EXPECT_EQ(uc3.z, uuc3.z);

    vector<3, short> s3(-4711, 0, 32767);
    short3 ss3 = cuda::cast<short3>(s3);
    EXPECT_EQ(s3.x, ss3.x);
    EXPECT_EQ(s3.y, ss3.y);
    EXPECT_EQ(s3.z, ss3.z);

    vector<3, unsigned short> us3(0, 4711, 65535);
    ushort3 uus3 = cuda::cast<ushort3>(us3);
    EXPECT_EQ(us3.x, uus3.x);
    EXPECT_EQ(us3.y, uus3.y);
    EXPECT_EQ(us3.z, uus3.z);

    vector<3, int> i3(-1024, 0, 2147483647);
    int3 ii3 = cuda::cast<int3>(i3);
    EXPECT_EQ(i3.x, ii3.x);
    EXPECT_EQ(i3.y, ii3.y);
    EXPECT_EQ(i3.z, ii3.z);

    vector<3, unsigned int> ui3(0, 1024, 4294967295);
    uint3 uui3 = cuda::cast<uint3>(ui3);
    EXPECT_EQ(ui3.x, uui3.x);
    EXPECT_EQ(ui3.y, uui3.y);
    EXPECT_EQ(ui3.z, uui3.z);

    vector<3, float> f3(0.0f, 4711.0f, 65535.0f);
    float3 ff3 = cuda::cast<float3>(f3);
    EXPECT_FLOAT_EQ(f3.x, ff3.x);
    EXPECT_FLOAT_EQ(f3.y, ff3.y);
    EXPECT_FLOAT_EQ(f3.z, ff3.z);

    // vec4 -----------------------------------------------

    vector<4, char> c4(-128, -96, 0, 127);
    char4 cc4 = cuda::cast<char4>(c4);
    EXPECT_EQ(c4.x, cc4.x);
    EXPECT_EQ(c4.y, cc4.y);
    EXPECT_EQ(c4.z, cc4.z);
    EXPECT_EQ(c4.w, cc4.w);

    vector<4, unsigned char> uc4(0, 32, 96, 255);
    uchar4 uuc4 = cuda::cast<uchar4>(uc4);
    EXPECT_EQ(uc4.x, uuc4.x);
    EXPECT_EQ(uc4.y, uuc4.y);
    EXPECT_EQ(uc4.z, uuc4.z);
    EXPECT_EQ(uc4.w, uuc4.w);

    vector<4, short> s4(-32768, -4711, 0, 32767);
    short4 ss4 = cuda::cast<short4>(s4);
    EXPECT_EQ(s4.x, ss4.x);
    EXPECT_EQ(s4.y, ss4.y);
    EXPECT_EQ(s4.z, ss4.z);
    EXPECT_EQ(s4.w, ss4.w);

    vector<4, unsigned short> us4(0, 1024, 4711, 65535);
    ushort4 uus4 = cuda::cast<ushort4>(us4);
    EXPECT_EQ(us4.x, uus4.x);
    EXPECT_EQ(us4.y, uus4.y);
    EXPECT_EQ(us4.z, uus4.z);
    EXPECT_EQ(us4.w, uus4.w);

    vector<4, int> i4(-2147483648, -1024, 0, 2147483647);
    int4 ii4 = cuda::cast<int4>(i4);
    EXPECT_EQ(i4.x, ii4.x);
    EXPECT_EQ(i4.y, ii4.y);
    EXPECT_EQ(i4.z, ii4.z);
    EXPECT_EQ(i4.w, ii4.w);

    vector<4, unsigned int> ui4(0, 1024, 4711, 4294967295);
    uint4 uui4 = cuda::cast<uint4>(ui4);
    EXPECT_EQ(ui4.x, uui4.x);
    EXPECT_EQ(ui4.y, uui4.y);
    EXPECT_EQ(ui4.z, uui4.z);
    EXPECT_EQ(ui4.w, uui4.w);

    vector<4, float> f4(0.0f, 4711.0f, 65535.0f, 4294967295.0f);
    float4 ff4 = cuda::cast<float4>(f4);
    EXPECT_FLOAT_EQ(f4.x, ff4.x);
    EXPECT_FLOAT_EQ(f4.y, ff4.y);
    EXPECT_FLOAT_EQ(f4.z, ff4.z);
    EXPECT_FLOAT_EQ(f4.w, ff4.w);
}

TEST(CastCU, NormToCuda)
{
    // vec2 -----------------------------------------------

    unsigned char arr8_2[2] = { 96, 255 };
    vector<2, unorm<8>> un8_2 = *reinterpret_cast<vector<2, unorm<8>>*>(arr8_2);
    uchar2 uun8_2 = cuda::cast<uchar2>(un8_2);
    EXPECT_EQ(arr8_2[0], uun8_2.x);
    EXPECT_EQ(arr8_2[1], uun8_2.y);

    unsigned short arr16_2[2] = { 4711, 65535 };
    vector<2, unorm<16>> un16_2 = *reinterpret_cast<vector<2, unorm<16>>*>(arr16_2);
    ushort2 uun16_2 = cuda::cast<ushort2>(un16_2);
    EXPECT_EQ(arr16_2[0], uun16_2.x);
    EXPECT_EQ(arr16_2[1], uun16_2.y);

    unsigned int arr32_2[2] = { 1024, 4294967295 };
    vector<2, unorm<32>> un32_2 = *reinterpret_cast<vector<2, unorm<32>>*>(arr32_2);
    uint2 uun32_2 = cuda::cast<uint2>(un32_2);
    EXPECT_EQ(arr32_2[0], uun32_2.x);
    EXPECT_EQ(arr32_2[1], uun32_2.y);

    // vec3 -----------------------------------------------

    unsigned char arr8_3[3] = { 0, 96, 255 };
    vector<3, unorm<8>> un8_3 = *reinterpret_cast<vector<3, unorm<8>>*>(arr8_3);
    uchar3 uun8_3 = cuda::cast<uchar3>(un8_3);
    EXPECT_EQ(arr8_3[0], uun8_3.x);
    EXPECT_EQ(arr8_3[1], uun8_3.y);
    EXPECT_EQ(arr8_3[2], uun8_3.z);

    unsigned short arr16_3[3] = { 0, 4711, 65535 };
    vector<3, unorm<16>> un16_3 = *reinterpret_cast<vector<3, unorm<16>>*>(arr16_3);
    ushort3 uun16_3 = cuda::cast<ushort3>(un16_3);
    EXPECT_EQ(arr16_3[0], uun16_3.x);
    EXPECT_EQ(arr16_3[1], uun16_3.y);
    EXPECT_EQ(arr16_3[2], uun16_3.z);

    unsigned int arr32_3[3] = { 0, 1024, 4294967295 };
    vector<3, unorm<32>> un32_3 = *reinterpret_cast<vector<3, unorm<32>>*>(arr32_3);
    uint3 uun32_3 = cuda::cast<uint3>(un32_3);
    EXPECT_EQ(arr32_3[0], uun32_3.x);
    EXPECT_EQ(arr32_3[1], uun32_3.y);
    EXPECT_EQ(arr32_3[2], uun32_3.z);

    // vec4 -----------------------------------------------

    unsigned char arr8_4[4] = { 0, 32, 96, 255 };
    vector<4, unorm<8>> un8_4 = *reinterpret_cast<vector<4, unorm<8>>*>(arr8_4);
    uchar4 uun8_4 = cuda::cast<uchar4>(un8_4);
    EXPECT_EQ(arr8_4[0], uun8_4.x);
    EXPECT_EQ(arr8_4[1], uun8_4.y);
    EXPECT_EQ(arr8_4[2], uun8_4.z);
    EXPECT_EQ(arr8_4[3], uun8_4.w);

    unsigned short arr16_4[4] = { 0, 1024, 4711, 65535 };
    vector<4, unorm<16>> un16_4 = *reinterpret_cast<vector<4, unorm<16>>*>(arr16_4);
    ushort4 uun16_4 = cuda::cast<ushort4>(un16_4);
    EXPECT_EQ(arr16_4[0], uun16_4.x);
    EXPECT_EQ(arr16_4[1], uun16_4.y);
    EXPECT_EQ(arr16_4[2], uun16_4.z);
    EXPECT_EQ(arr16_4[3], uun16_4.w);

    unsigned int arr32_4[4] = { 0, 1024, 4711, 4294967295 };
    vector<4, unorm<32>> un32_4 = *reinterpret_cast<vector<4, unorm<32>>*>(arr32_4);
    uint4 uun32_4 = cuda::cast<uint4>(un32_4);
    EXPECT_EQ(arr32_4[0], uun32_4.x);
    EXPECT_EQ(arr32_4[1], uun32_4.y);
    EXPECT_EQ(arr32_4[2], uun32_4.z);
    EXPECT_EQ(arr32_4[3], uun32_4.w);
}
