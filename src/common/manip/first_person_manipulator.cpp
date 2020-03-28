// This file is distributed under the MIT license.
// See the LICENSE file for details.

#ifndef NDEBUG
#include <iomanip>
#include <iostream>
#endif

#include <visionaray/math/axis.h>
#include <visionaray/math/forward.h>
#include <visionaray/math/matrix.h>
#include <visionaray/math/vector.h>
#include <visionaray/pinhole_camera.h>

#include "first_person_manipulator.h"
#include "../input/mouse_event.h"

using namespace visionaray;

namespace mouse = visionaray::mouse;


first_person_manipulator::first_person_manipulator(
        pinhole_camera& cam,
        mouse::buttons buttons,
        action_key_map const& action_keys
        )
    : camera_manipulator(cam)
    , buttons_(buttons)
    , modifiers_(keyboard::key_modifiers::NoKey)
    , down_modifiers_(keyboard::key_modifiers::NoKey)
    , dragging_(false)
    , action_keys_(action_keys)
{
}


first_person_manipulator::~first_person_manipulator()
{
}


void first_person_manipulator::handle_mouse_down(visionaray::mouse_event const& event)
{

    bool buttons   = event.buttons() & buttons_;
    bool modifiers = (modifiers_ == keyboard::NoKey && event.modifiers() == keyboard::NoKey)
                   || event.modifiers()  & modifiers_;

    if (!dragging_ && buttons && modifiers)
    {
        dragging_ = true;
        last_pos_ = event.pos();
        down_modifiers_ = event.modifiers();
    }

    camera_manipulator::handle_mouse_down(event);

}


void first_person_manipulator::handle_mouse_up(visionaray::mouse_event const& event)
{

    dragging_ = false;
    down_modifiers_ = keyboard::NoKey;

    camera_manipulator::handle_mouse_up(event);

}

void first_person_manipulator::handle_key_press(visionaray::key_event const& event)
{
    vec3 step_dir(0.0f);
    vec3 view_dir = normalize( camera_.eye() - camera_.center() );

    auto it = action_keys_.find(event.key());
    if (it != action_keys_.end())
    { 
        if (it->second == Left)
        {
            step_dir = cross(view_dir, camera_.up());
            step_dir.y = 0.0f;
            step_dir = normalize(step_dir);
        }

        if (it->second == Forward)
        {
            step_dir = -view_dir;
            step_dir.y = 0.0f;
            step_dir = normalize(step_dir);
        }

        if (it->second == Right)
        {
            step_dir = -cross(view_dir, camera_.up());
            step_dir.y = 0.0f;
            step_dir = normalize(step_dir);
        }

        if (it->second == Backward)
        {
            step_dir = view_dir;
            step_dir.y = 0.0f;
            step_dir = normalize(step_dir);
        }

        if (it->second == Down)
        {
            step_dir.y = -1.0f;
            step_dir = normalize(step_dir);
        }

        if (it->second == Up)
        {
            step_dir.y = 1.0f;
            step_dir = normalize(step_dir);
        }

    //  step_dir *= scale_;

        camera_.look_at(camera_.eye() + step_dir, camera_.center() + step_dir, camera_.up());
    }

    camera_manipulator::handle_key_press(event);

}

void first_person_manipulator::handle_mouse_move(visionaray::mouse_event const& event)
{

    bool buttons   = event.buttons() & buttons_;
    bool modifiers = (modifiers_ == keyboard::NoKey && down_modifiers_ == keyboard::NoKey)
                   || down_modifiers_ & modifiers_;

    if (dragging_ && buttons && modifiers)
    {
        auto w  =  camera_.get_viewport().w;
        auto h  =  camera_.get_viewport().h;
        auto dx =  static_cast<float>(last_pos_.x - event.pos().x) / w;
        auto dy =  static_cast<float>(last_pos_.y - event.pos().y) / h;
        dx *= constants::pi<float>();
        dy *= constants::pi_over_two<float>();

        vec3 axis = normalize(cross(camera_.up(), camera_.eye() - camera_.center()));
        mat4 xrot = mat4::rotation(axis, dy);
        mat4 yrot = mat4::rotation(to_vector(cartesian_axis<3>(cartesian_axis<3>::Y)), dx);

        mat4 rot = xrot * yrot;
        vec3 center = camera_.center() - camera_.eye();
        vec3 ct = (rot * vec4(center, 1.0f)).xyz();

        camera_.look_at(camera_.eye(), ct + camera_.eye(), camera_.up());
        last_pos_ = event.pos();
    }

    camera_manipulator::handle_mouse_move(event);

}
