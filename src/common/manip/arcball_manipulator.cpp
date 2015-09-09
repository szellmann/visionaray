// This file is distributed under the MIT license.
// See the LICENSE file for details.

#ifndef NDEBUG
#include <iomanip>
#include <iostream>
#endif

#include <visionaray/camera.h>

#include "arcball_manipulator.h"
#include "../input/mouse.h"


using namespace visionaray;

namespace mouse = visionaray::mouse;


arcball_manipulator::arcball_manipulator(camera& cam, mouse::buttons buttons)
    : camera_manipulator(cam)
    , buttons_(buttons)
    , dragging_(false)
{
}


arcball_manipulator::~arcball_manipulator()
{
}


void arcball_manipulator::handle_mouse_down(visionaray::mouse_event const& event)
{

    if (!dragging_)
    {
        dragging_ = true;
        ball_.down_pos = ball_.project(
                event.get_pos().x,
                event.get_pos().y,
                camera_.get_viewport()
                );
        ball_.down_rotation = ball_.rotation;
    }

    camera_manipulator::handle_mouse_down(event);

}


void arcball_manipulator::handle_mouse_up(visionaray::mouse_event const& event)
{

    dragging_ = false;
    camera_manipulator::handle_mouse_up(event);

}


void arcball_manipulator::handle_mouse_move(visionaray::mouse_event const& event)
{

    if (dragging_ && event.get_buttons() & buttons_)
    {

        // rotation

        vec3 curr_pos = ball_.project(
                event.get_pos().x,
                event.get_pos().y,
                camera_.get_viewport()
                );
        ball_.rotation = quat::rotation(ball_.down_pos, curr_pos) * ball_.down_rotation;

        if (true)
        {

            // view transform

            mat4 rotation_matrix = rotation(conjugate(ball_.rotation));

            vec4 eye4(0, 0, camera_.distance(), 1.0);
            eye4 = rotation_matrix * eye4;
            vec3 eye = vec3(eye4[0], eye4[1], eye4[2]);
            eye += camera_.center();

            vec4 up4 = rotation_matrix(1);
            vec3 up(up4[0], up4[1], up4[2]);

            camera_.look_at(eye, camera_.center(), up);

        }
        else
        {

            // model transform

            mat4 model = rotation(ball_.rotation);
            VSNRAY_UNUSED(model);
            //camera_.set_model_matrix(model);

        }

    }

    camera_manipulator::handle_mouse_move(event);

}
