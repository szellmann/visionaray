// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <visionaray/detail/macros.h>
#include <visionaray/pinhole_camera.h>

#include "camera_manipulator.h"
#include "../input/mouse.h"
#include "../input/key_event.h"
#include "../input/keyboard.h"

namespace visionaray
{

camera_manipulator::camera_manipulator(pinhole_camera& cam)
    : camera_(cam)
{
}

camera_manipulator::~camera_manipulator()
{
}

void camera_manipulator::handle_key_press(key_event const& event)
{
    VSNRAY_UNUSED(event);
}

void camera_manipulator::handle_key_release(key_event const& event)
{
    VSNRAY_UNUSED(event);
}

void camera_manipulator::handle_mouse_down(mouse_event const& event)
{
    VSNRAY_UNUSED(event);
}

void camera_manipulator::handle_mouse_up(mouse_event const& event)
{
    VSNRAY_UNUSED(event);
}

void camera_manipulator::handle_mouse_click(mouse_event const& event)
{
    VSNRAY_UNUSED(event);
}

void camera_manipulator::handle_mouse_move(mouse_event const& event)
{
    VSNRAY_UNUSED(event);
}

void camera_manipulator::handle_space_mouse_move(space_mouse_event const& event)
{
    VSNRAY_UNUSED(event);
}

} // visionaray
