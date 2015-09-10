// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_ARCBALL_MANIPULATOR_H
#define VSNRAY_ARCBALL_MANIPULATOR_H

#include <visionaray/math/math.h>

#include "../input/keyboard.h"
#include "../input/mouse.h"
#include "arcball.h"
#include "camera_manipulator.h"


namespace visionaray
{

class camera;
class mouse_event;

class arcball_manipulator : public camera_manipulator
{
public:

    arcball_manipulator(
            camera& cam,
            mouse::buttons buttons,
            keyboard::key_modifiers modifiers = keyboard::NoKey
            );
   ~arcball_manipulator();

    void handle_mouse_down(mouse_event const& event);
    void handle_mouse_up(mouse_event const& event);
    void handle_mouse_move(mouse_event const& event);

private:

    mouse::buttons buttons_;
    keyboard::key_modifiers modifiers_;

    keyboard::key_modifiers down_modifiers_;

    bool dragging_;

    arcball ball_;

};

} // visionaray

#endif // VSNRAY_ARCBALL_MANIPULATOR_H
