// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cstdint>
#include <cstring> // memcpy

#include <visionaray/math/math.h>

#include <gtest/gtest.h>

using namespace visionaray;

TEST(Unorm, Initialization)
{
    // Some convenience -----------------------------------

    static const uint8_t  max8  = numeric_limits<uint8_t>::max();
    static const uint16_t max16 = numeric_limits<uint16_t>::max();
    static const uint32_t max32 = numeric_limits<uint32_t>::max();

    static const uint8_t  low8  = numeric_limits<uint8_t>::lowest();
    static const uint16_t low16 = numeric_limits<uint16_t>::lowest();
    static const uint32_t low32 = numeric_limits<uint32_t>::lowest();


    // Test initialization --------------------------------

    unorm< 8> un8;
    unorm<16> un16;
    unorm<32> un32;
    unorm< 8> un8s[4];
    unorm<16> un16s[4];
    unorm<32> un32s[4];

    // init with 0

    un8  = 0.0f;
    un16 = 0.0f;
    un32 = 0.0f;

    EXPECT_EQ(uint8_t(un8),   low8);
    EXPECT_EQ(uint16_t(un16), low16);
    EXPECT_EQ(uint32_t(un32), low32);

    EXPECT_FLOAT_EQ(float(un8),  0.0f);
    EXPECT_FLOAT_EQ(float(un16), 0.0f);
    EXPECT_FLOAT_EQ(float(un32), 0.0f);


    // init with 1

    un8  = 1.0f;
    un16 = 1.0f;
    un32 = 1.0f;

    EXPECT_EQ(uint8_t(un8),   max8);
    EXPECT_EQ(uint16_t(un16), max16);
    EXPECT_EQ(uint32_t(un32), max32);

    EXPECT_FLOAT_EQ(float(un8),  1.0f);
    EXPECT_FLOAT_EQ(float(un16), 1.0f);
    EXPECT_FLOAT_EQ(float(un32), 1.0f);


    // init with byte arrays

    uint8_t arr8[]      = { 0, uint8_t( max8  / 2), max8,  max8 };
    uint16_t arr16[]    = { 0, uint16_t(max16 / 2), max16, max16 };
    uint32_t arr32[]    = { 0, uint32_t(max32 / 2), max32, max32 };

    std::memcpy(un8s,  arr8,  sizeof(arr8));
    std::memcpy(un16s, arr16, sizeof(arr16));
    std::memcpy(un32s, arr32, sizeof(arr32));

    for (int i = 0; i < 4; ++i)
    {
        EXPECT_EQ(uint8_t(un8s[i]),   arr8[i]);
        EXPECT_EQ(uint16_t(un16s[i]), arr16[i]);
        EXPECT_EQ(uint32_t(un32s[i]), arr32[i]);

        EXPECT_FLOAT_EQ(float(un8s[i]),  static_cast<float>(arr8[i])  / max8);
        EXPECT_FLOAT_EQ(float(un16s[i]), static_cast<float>(arr16[i]) / max16);
        EXPECT_FLOAT_EQ(float(un32s[i]), static_cast<float>(arr32[i]) / max32);
    }
}

TEST(Unorm, Comparisons)
{
    unorm< 8> a;
    unorm< 8> b;

    a = 0.5f;
    b = 1.0f;

    EXPECT_FALSE(a == b);
    EXPECT_TRUE( a != b);
    EXPECT_TRUE( a  < b);
    EXPECT_TRUE( a <= b);
    EXPECT_FALSE(a  > b);
    EXPECT_FALSE(a >= b);

    a = 1.0f;
    b = 0.5f;

    EXPECT_FALSE(a == b);
    EXPECT_TRUE( a != b);
    EXPECT_FALSE(a  < b);
    EXPECT_FALSE(a <= b);
    EXPECT_TRUE( a  > b);
    EXPECT_TRUE( a >= b);

    a = 1.0f;
    b = 1.0f;

    EXPECT_TRUE( a == b);
    EXPECT_FALSE(a != b);
    EXPECT_FALSE(a  < b);
    EXPECT_TRUE( a <= b);
    EXPECT_FALSE(a  > b);
    EXPECT_TRUE( a >= b);
}
