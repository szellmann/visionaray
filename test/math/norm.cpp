// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cstdint>
#include <cstring> // memcpy

#include <visionaray/math/math.h>

#include <gtest/gtest.h>

using namespace visionaray;

TEST(Norm, Unsigned)
{
    // Test initialization --------------------------------

    unorm< 8> un8;
    unorm<16> un16;
    unorm< 8> un8s[4];
    unorm<16> un16s[4];

    // init with 0

    un8 = 0.0f;

    EXPECT_EQ(uint8_t(un8),   numeric_limits<uint8_t>::lowest());
    EXPECT_EQ(uint16_t(un16), numeric_limits<uint16_t>::lowest());

    EXPECT_FLOAT_EQ(float(un8),  0.0f);
    EXPECT_FLOAT_EQ(float(un16), 0.0f);


    // init with 1

    un8  = 1.0f;
    un16 = 1.0f;

    EXPECT_EQ(uint8_t(un8),   numeric_limits<uint8_t>::max());
    EXPECT_EQ(uint16_t(un16), numeric_limits<uint16_t>::max());

    EXPECT_FLOAT_EQ(float(un8),  1.0f);
    EXPECT_FLOAT_EQ(float(un16), 1.0f);


    // init with byte arrays

    uint8_t arr8[]      = { 0,   128,   255,   255 };
    uint16_t arr16[]    = { 0, 32767, 65535, 65535 };

    std::memcpy(un8s,  arr8,  4 * sizeof(uint8_t));
    std::memcpy(un16s, arr16, 4 * sizeof(uint16_t));

    for (int i = 0; i < 4; ++i)
    {
        EXPECT_EQ(uint8_t(un8s[i]),   arr8[i]);
        EXPECT_EQ(uint16_t(un16s[i]), arr16[i]);

        EXPECT_FLOAT_EQ(float(un8s[i]),  static_cast<float>(arr8[i])  / numeric_limits<uint8_t>::max());
        EXPECT_FLOAT_EQ(float(un16s[i]), static_cast<float>(arr16[i]) / numeric_limits<uint16_t>::max());
    }
}
