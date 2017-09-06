// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_MANIP_PAN_MANIPULATOR_H
#define VSNRAY_COMMON_MANIP_PAN_MANIPULATOR_H 1

#include "../input/keyboard.h"
#include "camera_manipulator.h"


namespace visionaray
{

namespace detail
{
class perspective_camera_base;
} // detail
class mouse_event;

class pan_manipulator : public camera_manipulator
{
public:

    pan_manipulator(
            detail::perspective_camera_base& cam,
            mouse::buttons buttons,
            keyboard::key_modifiers modifiers = keyboard::NoKey
            );
   ~pan_manipulator();

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

#endif // VSNRAY_COMMON_MANIP_PAN_MANIPULATOR_H
