// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cstddef>
#include <sstream>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include <visionaray/math/serialization.h>

#include <gtest/gtest.h>

using namespace visionaray;


TEST(Serialization, AABB)
{
    // use as input
    basic_aabb<int>    dsti;
    basic_aabb<float>  dstf;
    basic_aabb<double> dstd;

    // test some invalid boxes
    basic_aabb<int>    srci;
    basic_aabb<float>  srcf;
    basic_aabb<double> srcd;

    srci.invalidate();
    srcf.invalidate();
    srcd.invalidate();

    std::stringstream sstream;
    boost::archive::text_oarchive oa(sstream);

    oa << srci;
    oa << srcf;
    oa << srcd;

    boost::archive::text_iarchive ia(sstream);

    ia >> dsti;
    ia >> dstf;
    ia >> dstd;

    EXPECT_EQ(dsti.min, srci.min);
    EXPECT_EQ(dsti.max, srci.max);
    EXPECT_FLOAT_EQ(dstf.min.x, srcf.min.x);
    EXPECT_FLOAT_EQ(dstf.min.y, srcf.min.y);
    EXPECT_FLOAT_EQ(dstf.min.z, srcf.min.z);
    EXPECT_FLOAT_EQ(dstf.max.x, srcf.max.x);
    EXPECT_FLOAT_EQ(dstf.max.y, srcf.max.y);
    EXPECT_FLOAT_EQ(dstf.max.z, srcf.max.z);
    EXPECT_DOUBLE_EQ(dstd.min.x, srcd.min.x);
    EXPECT_DOUBLE_EQ(dstd.min.y, srcd.min.y);
    EXPECT_DOUBLE_EQ(dstd.min.z, srcd.min.z);
    EXPECT_DOUBLE_EQ(dstd.max.x, srcd.max.x);
    EXPECT_DOUBLE_EQ(dstd.max.y, srcd.max.y);
    EXPECT_DOUBLE_EQ(dstd.max.z, srcd.max.z);
}

TEST(Serialization, Matrix)
{
    matrix<3, 3, float>  dst3f;
    matrix<3, 3, double> dst3d;
    matrix<4, 4, float>  dst4f;
    matrix<4, 4, double> dst4d;

    matrix<3, 3, float>  src3f(
            0.0f, 1.0f, 2.0f,
            3.0f, 4.0f, 5.0f,
            6.0f, 7.0f, 8.0f
            );
    matrix<3, 3, double> src3d(
            0.0, 1.0, 2.0,
            3.0, 4.0, 5.0,
            6.0, 7.0, 8.0
            );
    matrix<4, 4, float> src4f(
             0.0f,  1.0f,  2.0f,  3.0f,
             4.0f,  5.0f,  6.0f,  7.0f,
             8.0f,  9.0f, 10.0f, 11.0f,
            12.0f, 13.0f, 14.0f, 15.0f
            );
    matrix<4, 4, double> src4d(
             0.0,  1.0,  2.0,  3.0,
             4.0,  5.0,  6.0,  7.0,
             8.0,  9.0, 10.0, 11.0,
            12.0, 13.0, 14.0, 15.0
            );

    std::stringstream sstream;
    boost::archive::text_oarchive oa(sstream);

    oa << src3f;
    oa << src3d;
    oa << src4f;
    oa << src4d;

    boost::archive::text_iarchive ia(sstream);

    ia >> dst3f;
    ia >> dst3d;
    ia >> dst4f;
    ia >> dst4d;

    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j < 4; ++j)
        {
            if (i < 3 && j < 3)
            {
                EXPECT_FLOAT_EQ(dst3f(i, j), src3f(i, j));
                EXPECT_DOUBLE_EQ(dst3d(i, j), src3d(i, j));
            }
            EXPECT_FLOAT_EQ(dst4f(i, j), src4f(i, j));
            EXPECT_DOUBLE_EQ(dst4d(i, j), src4d(i, j));
        }
    }
}

TEST(Serialization, RectangleXYWH)
{
    rectangle<xywh_layout<int>, int>       dsti;
    rectangle<xywh_layout<float>, float>   dstf;
    rectangle<xywh_layout<double>, double> dstd;

    rectangle<xywh_layout<int>, int>       srci(0, 0, 1024, 768);
    rectangle<xywh_layout<float>, float>   srcf(-3.14f, -314.15f, 6.28f, 628.31f);
    rectangle<xywh_layout<double>, double> srcd(700.0, 700.0, 0.0, 0.0);

    std::stringstream sstream;
    boost::archive::text_oarchive oa(sstream);

    oa << srci;
    oa << srcf;
    oa << srcd;

    boost::archive::text_iarchive ia(sstream);

    ia >> dsti;
    ia >> dstf;
    ia >> dstd;

    EXPECT_EQ(dsti.x, srci.x);
    EXPECT_EQ(dsti.y, srci.y);
    EXPECT_EQ(dsti.w, srci.w);
    EXPECT_EQ(dsti.h, srci.h);

    EXPECT_FLOAT_EQ(dstf.x, srcf.x);
    EXPECT_FLOAT_EQ(dstf.y, srcf.y);
    EXPECT_FLOAT_EQ(dstf.w, srcf.w);
    EXPECT_FLOAT_EQ(dstf.h, srcf.h);

    EXPECT_DOUBLE_EQ(dstd.x, srcd.x);
    EXPECT_DOUBLE_EQ(dstd.y, srcd.y);
    EXPECT_DOUBLE_EQ(dstd.w, srcd.w);
    EXPECT_DOUBLE_EQ(dstd.h, srcd.h);
}

TEST(Serialization, Vector)
{
    float f1[] = { 3.14f };
    float f2[] = { 3.14f, 3.15f };
    float f3[] = { 3.14f, 3.15f, 3.16f };
    float f4[] = { 3.14f, 3.15f, 3.16f, 3.17f };
    float f5[] = { 3.14f, 3.15f, 3.16f, 3.17f, 3.18f };
    float f6[] = { 3.14f, 3.15f, 3.16f, 3.17f, 3.18f, 3.19f };
    float f7[] = { 3.14f, 3.15f, 3.16f, 3.17f, 3.18f, 3.19f, 3.20f };

    vector<1, float> src1f(f1);
    vector<2, float> src2f(f2);
    vector<3, float> src3f(f3);
    vector<4, float> src4f(f4);
    vector<5, float> src5f(f5);
    vector<6, float> src6f(f6);
    vector<7, float> src7f(f7);

    vector<1, float> dst1f;
    vector<2, float> dst2f;
    vector<3, float> dst3f;
    vector<4, float> dst4f;
    vector<5, float> dst5f;
    vector<6, float> dst6f;
    vector<7, float> dst7f;

    std::stringstream sstream;
    boost::archive::text_oarchive oa(sstream);

    oa << src1f;
    oa << src2f;
    oa << src3f;
    oa << src4f;
    oa << src5f;
    oa << src6f;
    oa << src7f;

    boost::archive::text_iarchive ia(sstream);

    ia >> dst1f;
    ia >> dst2f;
    ia >> dst3f;
    ia >> dst4f;
    ia >> dst5f;
    ia >> dst6f;
    ia >> dst7f;

    EXPECT_FLOAT_EQ(dst1f[0], src1f[0]);

    EXPECT_FLOAT_EQ(dst2f[0], src2f[0]);
    EXPECT_FLOAT_EQ(dst2f[1], src2f[1]);

    EXPECT_FLOAT_EQ(dst3f[0], src3f[0]);
    EXPECT_FLOAT_EQ(dst3f[1], src3f[1]);
    EXPECT_FLOAT_EQ(dst3f[2], src3f[2]);

    EXPECT_FLOAT_EQ(dst4f[0], src4f[0]);
    EXPECT_FLOAT_EQ(dst4f[1], src4f[1]);
    EXPECT_FLOAT_EQ(dst4f[2], src4f[2]);
    EXPECT_FLOAT_EQ(dst4f[3], src4f[3]);

    EXPECT_FLOAT_EQ(dst5f[0], src5f[0]);
    EXPECT_FLOAT_EQ(dst5f[1], src5f[1]);
    EXPECT_FLOAT_EQ(dst5f[2], src5f[2]);
    EXPECT_FLOAT_EQ(dst5f[3], src5f[3]);
    EXPECT_FLOAT_EQ(dst5f[4], src5f[4]);

    EXPECT_FLOAT_EQ(dst6f[0], src6f[0]);
    EXPECT_FLOAT_EQ(dst6f[1], src6f[1]);
    EXPECT_FLOAT_EQ(dst6f[2], src6f[2]);
    EXPECT_FLOAT_EQ(dst6f[3], src6f[3]);
    EXPECT_FLOAT_EQ(dst6f[4], src6f[4]);
    EXPECT_FLOAT_EQ(dst6f[5], src6f[5]);

    EXPECT_FLOAT_EQ(dst7f[0], src7f[0]);
    EXPECT_FLOAT_EQ(dst7f[1], src7f[1]);
    EXPECT_FLOAT_EQ(dst7f[2], src7f[2]);
    EXPECT_FLOAT_EQ(dst7f[3], src7f[3]);
    EXPECT_FLOAT_EQ(dst7f[4], src7f[4]);
    EXPECT_FLOAT_EQ(dst7f[5], src7f[5]);
    EXPECT_FLOAT_EQ(dst7f[6], src7f[6]);
}
