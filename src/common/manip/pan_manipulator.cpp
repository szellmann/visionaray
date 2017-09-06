// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <visionaray/detail/perspective_camera_base.h>

#include "pan_manipulator.h"


using namespace visionaray;


pan_manipulator::pan_manipulator(
        detail::perspective_camera_base& cam,
        mouse::buttons buttons,
        keyboard::key_modifiers modifiers
        )
    : camera_manipulator(cam)
    , buttons_(buttons)
    , modifiers_(modifiers)
    , dragging_(false)
    , down_modifiers_(keyboard::key_modifiers::NoKey)
{
}


pan_manipulator::~pan_manipulator()
{
}


void pan_manipulator::handle_mouse_down(visionaray::mouse_event const& event)
{

    bool buttons   = event.get_buttons() & buttons_;
    bool modifiers = (modifiers_ == keyboard::NoKey && event.get_modifiers() == keyboard::NoKey)
                   || event.get_modifiers()  & modifiers_;

    if (!dragging_ && buttons && modifiers)
    {

        dragging_ = true;

        // TODO: do this in base?
        last_pos_ = event.get_pos();
        down_modifiers_ = event.get_modifiers();
    }

    camera_manipulator::handle_mouse_down(event);

}


void pan_manipulator::handle_mouse_up(visionaray::mouse_event const& event)
{

    dragging_ = false;
    down_modifiers_ = keyboard::NoKey;

    camera_manipulator::handle_mouse_up(event);

}


void pan_manipulator::handle_mouse_move(visionaray::mouse_event const& event)
{

    bool buttons   = event.get_buttons() & buttons_;
    bool modifiers = (modifiers_ == keyboard::NoKey && down_modifiers_ == keyboard::NoKey) || down_modifiers_ & modifiers_;

    if (dragging_ && buttons && modifiers)
    {

        auto w  =  camera_.get_viewport().w;
        auto h  =  camera_.get_viewport().h;
        auto dx =  static_cast<float>(last_pos_.x - event.get_pos().x) / w;
        auto dy = -static_cast<float>(last_pos_.y - event.get_pos().y) / h;
        auto s  = 2.0f * camera_.distance();
        auto zaxis = normalize( camera_.eye() - camera_.center() );
        auto yaxis = camera_.up();
        auto xaxis = cross( yaxis, zaxis );
        vec3 d  =  (dx * s) * xaxis + (dy * s) * yaxis;

        camera_.look_at( camera_.eye() + d, camera_.center() + d, camera_.up() );

        last_pos_ = event.get_pos();

    }

    camera_manipulator::handle_mouse_move(event);

}
