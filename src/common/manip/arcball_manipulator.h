// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_ARCBALL_MANIPULATOR_H
#define VSNRAY_ARCBALL_MANIPULATOR_H

#include <visionaray/math/math.h>

#include "camera_manipulator.h"
#include "../input/mouse.h"


namespace visionaray
{

class camera;
class mouse_event;

class arcball_manipulator : public camera_manipulator
{
public:

    arcball_manipulator(camera& cam, mouse::buttons buttons);
   ~arcball_manipulator();

    void handle_mouse_down(mouse_event const& event);
    void handle_mouse_up(mouse_event const& event);
    void handle_mouse_move(mouse_event const& event);

private:

    mouse::buttons buttons_;

    float radius_;

    bool  dragging_;

    vec3 down_pos_;
    quat rotation_;
    quat down_rotation_;

    vec3 to_sphere_coords(int x, int y);

};

} // visionaray

#endif // VSNRAY_ARCBALL_MANIPULATOR_H
