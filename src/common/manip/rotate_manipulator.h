// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_MANIP_ROTATE_MANIPULATOR_H
#define VSNRAY_COMMON_MANIP_ROTATE_MANIPULATOR_H 1

#include "../input/mouse.h"
#include "arcball.h"
#include "model_manipulator.h"

namespace visionaray
{

// TODO: move to library?
struct disci
{
    vec2i center;
    int radius;

    bool contains(vec2i v) const
    {
        v -= center;
        return length(v) <= radius;
    }
};

class rotate_manipulator : public model_manipulator
{
public:

    rotate_manipulator(
            camera const& cam,
            mat4& model_matrix,
            vec3 size,
            mouse::buttons buttons
            );

    void render();

private:

    bool handle_mouse_down(mouse_event const& event);
    bool handle_mouse_up(mouse_event const& event);
    bool handle_mouse_move(mouse_event const& event);

    int select_from_mouse_pointer(mouse_event const& event);

    disci bounding_disc();
    recti bounding_rect();

    mouse::buttons buttons_;
    mouse::pos down_pos_;

    int hovered_ = -1;
    int selected_ = -1;

    bool dragging_ = false;
    bool mouse_over_ = false;

    arcball ball_;

};

} // visionaray

#endif // VSNRAY_COMMON_MANIP_ROTATE_MANIPULATOR_H
