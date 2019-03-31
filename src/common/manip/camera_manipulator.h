// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_MANIP_CAMERA_MANIPULATOR_H
#define VSNRAY_COMMON_MANIP_CAMERA_MANIPULATOR_H 1

#include "../input/key_event.h"
#include "../input/mouse_event.h"
#include "../input/space_mouse.h"

namespace visionaray
{

class pinhole_camera;

class camera_manipulator
{
public:

    camera_manipulator(pinhole_camera& cam);
    virtual ~camera_manipulator();

    virtual void handle_key_press(key_event const& event);
    virtual void handle_key_release(key_event const& event);

    virtual void handle_mouse_down(mouse_event const& event);
    virtual void handle_mouse_up(mouse_event const& event);
    virtual void handle_mouse_click(mouse_event const& event);
    virtual void handle_mouse_move(mouse_event const& event);

    virtual void handle_space_mouse_move(space_mouse_event const& event);

protected:

    pinhole_camera& camera_;

};

} // visionaray

#endif // VSNRAY_COMMON_MANIP_CAMERA_MANIPULATOR_H
