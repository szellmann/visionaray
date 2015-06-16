// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_PAN_MANIPULATOR_H
#define VSNRAY_PAN_MANIPULATOR_H

#include "camera_manipulator.h"


namespace visionaray
{

class camera;
class mouse_event;

class pan_manipulator : public camera_manipulator
{
public:

    pan_manipulator(camera& cam, mouse::buttons buttons);
   ~pan_manipulator();

    void handle_mouse_down(mouse_event const& event);
    void handle_mouse_up(mouse_event const& event);
    void handle_mouse_move(mouse_event const& event);

private:

    mouse::buttons buttons_;

    bool dragging_;

    mouse::pos last_pos_;

};

} // visionaray

#endif // VSNRAY_PAN_MANIPULATOR_H
