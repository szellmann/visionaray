// This file is distributed under the MIT license.
// See the LICENSE file for details

#include <visionaray/math/math.h>
#include <visionaray/result_record.h>
#include <visionaray/simple_buffer_rt.h>
#include <visionaray/scheduler.h>

#include <gtest/gtest.h>

using namespace visionaray;


//-------------------------------------------------------------------------------------------------
// Simple kernels that store result in render target
// Test if pixel access works as expected
//

// kernel returns a color
template <typename R, typename RT>
void test_pixel_access_color(
        R       /* */,
        RT&     rt
        )
{
    // dummies
    mat4 mv = mat4::identity();
    mat4 pr = mat4::identity();

    auto sparams = make_sched_params(pixel_sampler::uniform_type{}, mv, pr, rt);

    simple_sched<ray> sched;

    sched.frame([&](R) -> typename RT::color_type
    {
        using C = typename RT::color_type;
        return C(0.5f);
    }, sparams);
}

// kernel returns a result record
template <typename R, typename RT>
void test_pixel_access_result_record(
        R       /* */,
        RT&     rt
        )
{
    using S = typename R::scalar_type;


    // dummies
    mat4 mv = mat4::identity();
    mat4 pr = mat4::identity();

    auto sparams = make_sched_params(pixel_sampler::uniform_type{}, mv, pr, rt);

    simple_sched<ray> sched;

    sched.frame([&](R) -> result_record<S>
    {
        result_record<S> result;
        result.color = typename result_record<S>::color_type(0.4f);
        return result;
    }, sparams);
}


//-------------------------------------------------------------------------------------------------
// Test if pixel access to render targets works
//

TEST(RenderTarget, PixelFormats)
{
    simple_buffer_rt<PF_RGB32F, PF_UNSPECIFIED> rt_RGB32F;
    rt_RGB32F.resize(1, 1);

    simple_buffer_rt<PF_RGBA32F, PF_UNSPECIFIED> rt_RGBA32F;
    rt_RGBA32F.resize(1, 1);


    // Color access, color: RGB32F, depth: UNSPECIFIED ------------------------

    test_pixel_access_color(ray(), rt_RGB32F);
    EXPECT_FLOAT_EQ(rt_RGB32F.color()[0].x, 0.5f);
    EXPECT_FLOAT_EQ(rt_RGB32F.color()[0].y, 0.5f);
    EXPECT_FLOAT_EQ(rt_RGB32F.color()[0].z, 0.5f);


    // Color access, color: RGBA32F, depth: UNSPECIFIED -----------------------

    test_pixel_access_color(ray(), rt_RGBA32F);
    EXPECT_FLOAT_EQ(rt_RGBA32F.color()[0].x, 0.5f);
    EXPECT_FLOAT_EQ(rt_RGBA32F.color()[0].y, 0.5f);
    EXPECT_FLOAT_EQ(rt_RGBA32F.color()[0].z, 0.5f);
    EXPECT_FLOAT_EQ(rt_RGBA32F.color()[0].w, 0.5f);


    // Result record, color: RGB32F, depth: UNSPECIFIED -----------------------

    test_pixel_access_result_record(ray(), rt_RGB32F);
    EXPECT_FLOAT_EQ(rt_RGB32F.color()[0].x, 0.16); // alpha multiplication!
    EXPECT_FLOAT_EQ(rt_RGB32F.color()[0].y, 0.16);
    EXPECT_FLOAT_EQ(rt_RGB32F.color()[0].z, 0.16);


    // Result record, color: RGB32F, depth: UNSPECIFIED -----------------------

    test_pixel_access_result_record(ray(), rt_RGBA32F);
    EXPECT_FLOAT_EQ(rt_RGBA32F.color()[0].x, 0.4f);
    EXPECT_FLOAT_EQ(rt_RGBA32F.color()[0].y, 0.4f);
    EXPECT_FLOAT_EQ(rt_RGBA32F.color()[0].z, 0.4f);
}
