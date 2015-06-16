// This file is distributed under the MIT license.
// See the LICENSE file for details.

#ifndef NDEBUG
#include <iomanip>
#include <iostream>
#endif

#include <visionaray/camera.h>

#include "zoom_manipulator.h"


using namespace visionaray;


zoom_manipulator::zoom_manipulator(camera& cam, mouse::buttons buttons)
    : camera_manipulator(cam)
    , buttons_(buttons)
    , dragging_(false)
{
}


zoom_manipulator::~zoom_manipulator()
{
}


void zoom_manipulator::handle_mouse_down(visionaray::mouse_event const& event)
{

    if (dragging_)
    {
        return;
    }

    dragging_ = true;

    // TODO: do this in base?
    last_pos_ = event.get_pos();

    camera_manipulator::handle_mouse_down(event);

}


void zoom_manipulator::handle_mouse_up(visionaray::mouse_event const& event)
{

    dragging_ = false;

    camera_manipulator::handle_mouse_up(event);

}


void zoom_manipulator::handle_mouse_move(visionaray::mouse_event const& event)
{

    if (!dragging_)
    {
        return;
    }


    if (event.get_buttons() & buttons_)
    {

//      float w  = camera_.get_viewport().w;
        float h  = camera_.get_viewport().h;
//      float dx =  static_cast<float>(last_pos_.x - event.get_pos().x) / w;
        float dy = -static_cast<float>(last_pos_.y - event.get_pos().y) / h;
        float s  = 2.0f * camera_.distance() * dy;

        vec3 dir = normalize( camera_.eye() - camera_.center() );

        camera_.look_at( camera_.eye() - dir * s, camera_.center(), camera_.up() );

        last_pos_ = event.get_pos();

    }

    camera_manipulator::handle_mouse_move(event);

}
