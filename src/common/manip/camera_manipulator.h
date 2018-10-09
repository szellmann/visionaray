// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_MANIP_CAMERA_MANIPULATOR_H
#define VSNRAY_COMMON_MANIP_CAMERA_MANIPULATOR_H 1

#include <visionaray/detail/macros.h>

#include "../input/key_event.h"
#include "../input/mouse_event.h"


namespace visionaray
{

class pinhole_camera;

class camera_manipulator
{
public:

    camera_manipulator(pinhole_camera& cam) : camera_(cam) {}
    virtual ~camera_manipulator() {}

    virtual void handle_key_press(key_event const& event)     { VSNRAY_UNUSED(event); }
    virtual void handle_key_release(key_event const& event)   { VSNRAY_UNUSED(event); }

    virtual void handle_mouse_down(mouse_event const& event)  { VSNRAY_UNUSED(event); }
    virtual void handle_mouse_up(mouse_event const& event)    { VSNRAY_UNUSED(event); }
    virtual void handle_mouse_click(mouse_event const& event) { VSNRAY_UNUSED(event); }
    virtual void handle_mouse_move(mouse_event const& event)  { VSNRAY_UNUSED(event); }

protected:

    pinhole_camera& camera_;

};


class fp_manipulator : public camera_manipulator
{
public:

    fp_manipulator(pinhole_camera& cam);
   ~fp_manipulator();

    void handle_key_press(key_event const& event);

};

} // visionaray

#endif // VSNRAY_COMMON_MANIP_CAMERA_MANIPULATOR_H
