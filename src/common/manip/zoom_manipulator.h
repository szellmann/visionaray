// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_MANIP_ZOOM_MANIPULATOR_H
#define VSNRAY_COMMON_MANIP_ZOOM_MANIPULATOR_H 1

#include "../input/keyboard.h"
#include "../input/mouse.h"
#include "camera_manipulator.h"


namespace visionaray
{

class pinhole_camera;
class mouse_event;

class zoom_manipulator : public camera_manipulator
{
public:

    zoom_manipulator(
            pinhole_camera& cam,
            mouse::buttons buttons,
            keyboard::key_modifiers modifiers = keyboard::NoKey
            );
    ~zoom_manipulator();

    void handle_mouse_down(mouse_event const& event);
    void handle_mouse_up(mouse_event const& event);
    void handle_mouse_move(mouse_event const& event);

private:

    mouse::buttons buttons_;
    keyboard::key_modifiers modifiers_;

    bool dragging_;

    mouse::pos last_pos_;
    keyboard::key_modifiers down_modifiers_;

};

} // visionaray

#endif // VSNRAY_COMMON_MANIP_ZOOM_MANIPULATOR_H
