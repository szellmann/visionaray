// This file is distributed under the MIT license.
// See the LICENSE file for details.

#ifndef NDEBUG
#include <iomanip>
#include <iostream>
#endif

#include <visionaray/math/axis.h>
#include <visionaray/pinhole_camera.h>

#include "first_person_manipulator.h"
#include "../input/mouse_event.h"


using namespace visionaray;

namespace mouse = visionaray::mouse;


first_person_manipulator::first_person_manipulator(
        pinhole_camera& cam,
        mouse::buttons buttons,
        keyboard::key_modifiers modifiers
        )
    : camera_manipulator(cam)
    , buttons_(buttons)
    , modifiers_(modifiers)
    , down_modifiers_(keyboard::key_modifiers::NoKey)
    , dragging_(false)
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

    // Left
    if (event.key() == keyboard::a || event.key() == keyboard::A)
    {
        step_dir = cross(view_dir, camera_.up());
        step_dir.y = 0.0f;
    }

    // Forward
    if (event.key() == keyboard::w || event.key() == keyboard::W)
    {
        step_dir = -view_dir;
        step_dir.y = 0.0f;
    }

    // Right
    if (event.key() == keyboard::d || event.key() == keyboard::D)
    {
        step_dir = -cross(view_dir, camera_.up());
        step_dir.y = 0.0f;
    }

    // Back
    if (event.key() == keyboard::s || event.key() == keyboard::S)
    {
        step_dir = view_dir;
        step_dir.y = 0.0f;
    }

    // Down
    if (event.key() == keyboard::f || event.key() == keyboard::F)
    {
        step_dir.y = -1.0f;
 
    }

    // Up
    if (event.key() == keyboard::r || event.key() == keyboard::R)
    {
        step_dir.y = 1.0f;
    }

    step_dir = normalize(step_dir);

//  step_dir *= scale_;

    camera_.look_at(camera_.eye() + step_dir, camera_.center() + step_dir, camera_.up());
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
