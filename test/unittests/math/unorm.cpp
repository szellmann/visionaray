// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cstdint>
#include <cstring> // memcpy

#include <visionaray/math/math.h>

#include <gtest/gtest.h>

using namespace visionaray;


//-------------------------------------------------------------------------------------------------
// Test unorm initialization (single values, arrays)
//

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


//-------------------------------------------------------------------------------------------------
// Test comparison operators
//

template <unsigned Bits>
void test_cmp()
{
    unorm<Bits> a;
    unorm<Bits> b;

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

TEST(Unorm, Comparisons)
{
    test_cmp< 8>();
    test_cmp<16>();
    test_cmp<32>();
}


//-------------------------------------------------------------------------------------------------
// Test numeric limits
// All representations (norm, int, float)
//

TEST(Unorm, NumericLimits)
{
    // Some convenience -----------------------------------

    static const uint8_t   uint8_max = numeric_limits<uint8_t>::max();
    static const uint16_t uint16_max = numeric_limits<uint16_t>::max();
    static const uint32_t uint32_max = numeric_limits<uint32_t>::max();

    static const uint8_t   uint8_low = numeric_limits<uint8_t>::lowest();
    static const uint16_t uint16_low = numeric_limits<uint16_t>::lowest();
    static const uint32_t uint32_low = numeric_limits<uint32_t>::lowest();

    static const uint8_t   uint8_min = numeric_limits<uint8_t>::min();
    static const uint16_t uint16_min = numeric_limits<uint16_t>::min();
    static const uint32_t uint32_min = numeric_limits<uint32_t>::min();


    // Normalized reprentation ----------------------------

    EXPECT_TRUE(numeric_limits<unorm< 8>>::max() == unorm< 8>(1.0f));
    EXPECT_TRUE(numeric_limits<unorm<16>>::max() == unorm<16>(1.0f));
    EXPECT_TRUE(numeric_limits<unorm<32>>::max() == unorm<32>(1.0f));

    EXPECT_TRUE(numeric_limits<unorm< 8>>::lowest() == unorm< 8>(0.0f));
    EXPECT_TRUE(numeric_limits<unorm<16>>::lowest() == unorm<16>(0.0f));
    EXPECT_TRUE(numeric_limits<unorm<32>>::lowest() == unorm<32>(0.0f));

    EXPECT_TRUE(numeric_limits<unorm< 8>>::min() == unorm< 8>(0.0f));
    EXPECT_TRUE(numeric_limits<unorm<16>>::min() == unorm<16>(0.0f));
    EXPECT_TRUE(numeric_limits<unorm<32>>::min() == unorm<32>(0.0f));


    // Integer representation -----------------------------

    EXPECT_EQ(static_cast< uint8_t>(numeric_limits<unorm< 8>>::max()), uint8_max);
    EXPECT_EQ(static_cast<uint16_t>(numeric_limits<unorm<16>>::max()), uint16_max);
    EXPECT_EQ(static_cast<uint32_t>(numeric_limits<unorm<32>>::max()), uint32_max);

    EXPECT_EQ(static_cast< uint8_t>(numeric_limits<unorm< 8>>::lowest()), uint8_low);
    EXPECT_EQ(static_cast<uint16_t>(numeric_limits<unorm<16>>::lowest()), uint16_low);
    EXPECT_EQ(static_cast<uint32_t>(numeric_limits<unorm<32>>::lowest()), uint32_low);

    EXPECT_EQ(static_cast< uint8_t>(numeric_limits<unorm< 8>>::min()), uint8_min);
    EXPECT_EQ(static_cast<uint16_t>(numeric_limits<unorm<16>>::min()), uint16_min);
    EXPECT_EQ(static_cast<uint32_t>(numeric_limits<unorm<32>>::min()), uint32_min);


    // Float representation -------------------------------

    EXPECT_FLOAT_EQ(static_cast<float>(numeric_limits<unorm< 8>>::max()),    1.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(numeric_limits<unorm<16>>::max()),    1.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(numeric_limits<unorm<32>>::max()),    1.0f);

    EXPECT_FLOAT_EQ(static_cast<float>(numeric_limits<unorm< 8>>::lowest()), 0.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(numeric_limits<unorm<16>>::lowest()), 0.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(numeric_limits<unorm<32>>::lowest()), 0.0f);

    EXPECT_FLOAT_EQ(static_cast<float>(numeric_limits<unorm< 8>>::min()),    0.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(numeric_limits<unorm<16>>::min()),    0.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(numeric_limits<unorm<32>>::min()),    0.0f);
}
