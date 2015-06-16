// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_ZOOM_MANIPULATOR_H
#define VSNRAY_ZOOM_MANIPULATOR_H

#include <visionaray/math/math.h>

#include "camera_manipulator.h"
#include "../input/mouse.h"


namespace visionaray
{

class camera;
class mouse_event;

class zoom_manipulator : public camera_manipulator
{
public:

    zoom_manipulator(camera& cam, mouse::buttons buttons);
   ~zoom_manipulator();

    void handle_mouse_down(mouse_event const& event);
    void handle_mouse_up(mouse_event const& event);
    void handle_mouse_move(mouse_event const& event);

private:

    mouse::buttons buttons_;

    bool dragging_;

    mouse::pos last_pos_;

};

} // visionaray

#endif // VSNRAY_ZOOM_MANIPULATOR_H
