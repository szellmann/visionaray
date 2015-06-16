// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <visionaray/camera.h>

#include "pan_manipulator.h"


using namespace visionaray;


pan_manipulator::pan_manipulator(camera& cam, mouse::buttons buttons)
    : camera_manipulator(cam)
    , buttons_(buttons)
    , dragging_(false)
{
}


pan_manipulator::~pan_manipulator()
{
}


void pan_manipulator::handle_mouse_down(visionaray::mouse_event const& event)
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


void pan_manipulator::handle_mouse_up(visionaray::mouse_event const& event)
{

    dragging_ = false;

    camera_manipulator::handle_mouse_up(event);

}


void pan_manipulator::handle_mouse_move(visionaray::mouse_event const& event)
{

    if (!dragging_)
    {
        return;
    }


    if (event.get_buttons() & buttons_)
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
