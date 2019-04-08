// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <visionaray/pinhole_camera.h>

#include "model_manipulator.h"

using namespace visionaray;

model_manipulator::model_manipulator(
        pinhole_camera const& cam,
        mat4& model_matrix,
        vec3 size
        )
    : camera_(cam)
    , model_matrix_(model_matrix)
    , size_(size)
    , active_(false)
{
}

model_manipulator::~model_manipulator()
{
}

void model_manipulator::set_active(bool active)
{
    active_ = active;
}

bool model_manipulator::active() const
{
    return active_;
}

void model_manipulator::render()
{
}

bool model_manipulator::handle_key_press(key_event const& event)
{
    VSNRAY_UNUSED(event);
    return false;
}

bool model_manipulator::handle_key_release(key_event const& event)
{
    VSNRAY_UNUSED(event);
    return false;
}

bool model_manipulator::handle_mouse_down(mouse_event const& event)
{
    VSNRAY_UNUSED(event);
    return false;
}

bool model_manipulator::handle_mouse_up(mouse_event const& event)
{
    VSNRAY_UNUSED(event);
    return false;
}

bool model_manipulator::handle_mouse_click(mouse_event const& event)
{
    VSNRAY_UNUSED(event);
    return false;
}

bool model_manipulator::handle_mouse_move(mouse_event const& event)
{
    VSNRAY_UNUSED(event);
    return false;
}

bool model_manipulator::handle_space_mouse_move(space_mouse_event const& event)
{
    VSNRAY_UNUSED(event);
    return false;
}

bool model_manipulator::handle_space_mouse_button_press(space_mouse_event const& event)
{
    VSNRAY_UNUSED(event);
    return false;
}

