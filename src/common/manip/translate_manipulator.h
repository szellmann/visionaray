// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_MANIP_TRANSLATE_MANIPULATOR_H
#define VSNRAY_COMMON_MANIP_TRANSLATE_MANIPULATOR_H 1

#include "model_manipulator.h"

namespace visionaray
{

class translate_manipulator : public model_manipulator
{
public:

    translate_manipulator(
            pinhole_camera const& cam,
            mat4& model_matrix,
            vec3 size,
            mouse::buttons buttons
            );

    void render();

private:

    bool handle_mouse_down(mouse_event const& event);
    bool handle_mouse_up(mouse_event const& event);
    bool handle_mouse_move(mouse_event const& event);

    bool handle_space_mouse_move(space_mouse_event const& event);

    int select_from_mouse_pointer(mouse_event const& event);

    vec3 to_sphere_coords(int x, int y);

    int hovered_ = -1;
    int selected_ = -1;

    bool dragging_ = false;

    mouse::buttons buttons_;
    mouse::pos down_pos_;

};

} // visionaray

#endif // VSNRAY_COMMON_MANIP_TRANSLATE_MANIPULATOR_H
