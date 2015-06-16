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


quat quat_from_sphere_coords(vec3 const& from, vec3 const& to)
{
    vec3 nfrom = normalize(from);
    vec3 nto   = normalize(to);

    return quat(dot(nfrom, nto), cross(nfrom, nto));
}


arcball_manipulator::arcball_manipulator(camera& cam, mouse::buttons buttons)
    : camera_manipulator(cam)
    , buttons_(buttons)
    , radius_(1.0f)
    , dragging_(false)
    , down_pos_(0.0f)
    , rotation_(quat::identity())
    , down_rotation_(quat::identity())
{
}


arcball_manipulator::~arcball_manipulator()
{
}


void arcball_manipulator::handle_mouse_down(visionaray::mouse_event const& event)
{

    if (dragging_)
    {
        return;
    }

    dragging_      = true;
    down_pos_      = to_sphere_coords(event.get_pos().x, event.get_pos().y);
    down_rotation_ = rotation_;

    camera_manipulator::handle_mouse_down(event);

}


void arcball_manipulator::handle_mouse_up(visionaray::mouse_event const& event)
{

    dragging_ = false;
    camera_manipulator::handle_mouse_up(event);

}


void arcball_manipulator::handle_mouse_move(visionaray::mouse_event const& event)
{

    if (!dragging_)
    {
        return;
    }


    if (event.get_buttons() & buttons_)
    {

        // rotation

        vec3 curr_pos = to_sphere_coords(event.get_pos().x, event.get_pos().y);
        rotation_ = quat_from_sphere_coords(down_pos_, curr_pos) * down_rotation_;

        if (true)
        {

            // view transform

            mat4 rotation_matrix = rotation(conjugate(rotation_));

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

            mat4 model = rotation(rotation_);
            VSNRAY_UNUSED(model);
            //camera_.set_model_matrix(model);

        }

    }

    camera_manipulator::handle_mouse_move(event);

}


vec3 arcball_manipulator::to_sphere_coords(int x, int y)
{

    vec3 v(0.0f);

    int width  = camera_.get_viewport().w;
    int height = camera_.get_viewport().h;

#if 0

    // trackball

    v[0] =  (x - 0.5f * width ) / width;
    v[1] = -(y - 0.5f * height) / height;

    vec2 tmp(v[0], v[1]);
    float d = normh2(tmp);
    float r2 = radius_ * radius_;

    if (d < radius_ * (1.0f / std::sqrt(2.0)))
    {
        v[2] = std::sqrt(r2 - d * d);
    }
    else
    {
        v[2] = r2 / (2.0f * d);
    }
#else

    //arcball

    v[0] =  (x - 0.5f * width ) / (radius_ * 0.5f * width );
    v[1] = -(y - 0.5f * height) / (radius_ * 0.5f * height);

    vec2 tmp(v[0], v[1]);
    float d = norm2(tmp);


    if (d > 1.0f)
    {
        float length = std::sqrt(d);

        v[0] /= length;
        v[1] /= length;
    }
    else
    {
        v[2] = std::sqrt(1.0f - d);
    }
#endif

    return v;

}
