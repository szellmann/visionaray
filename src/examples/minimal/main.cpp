// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <visionaray/camera.h>
#include <visionaray/scheduler.h>
#include <visionaray/simple_buffer_rt.h>

int main()
{

// 1.)
    // Visionaray's main namespace.
    using namespace visionaray;

    // Define some convenience types.
    typedef basic_ray<float>       ray_type;
    typedef simple_sched<ray_type> sched_type;
    typedef vector<4, float>       color_type;

// 2.)
    int w = 100;
    int h = 100;

    // A simple pinhole camera.
    camera cam;
    cam.set_viewport(0, 0, w, h);
    float aspect = w / static_cast<float>(h);

    // Set camera projection like with gluPerspective:
    // 1. parameter: field of view angle in radians
    // 2. parameter: aspect ratio
    // 3. parameter: near and far clipping planes
    cam.perspective
    (
        45.0f * constants::pi<float>() / 180.0f,
        aspect,
        0.001f, 1000.0f
    );

// 3.)
    // The simple buffer render target provides storage
    // for the color buffer and depth buffer.
    simple_buffer_rt<PF_RGBA32F, PF_UNSPECIFIED> render_target;
    render_target.resize(100, 100);

// 4.)
    // Scheduler instance, probably a class member
    // in a real-world example.
    sched_type sched;
    auto sparams = make_sched_params
    (
        cam.get_view_matrix(),
        cam.get_proj_matrix(),
        cam.get_viewport(),
        render_target
    );

// 5.)
    // In this example, the kernel is passed to the
    // scheduler as a lambda. It specifies the code
    // path that a single ray or a ray packet executes.
    sched.frame([=](ray_type r) -> color_type
    {
        return color_type(1.0, 1.0, 1.0, 1.0);
    },
    sparams);
}
