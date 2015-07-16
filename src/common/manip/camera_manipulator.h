// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_CAMERA_MANIPULATOR_H
#define VSNRAY_CAMERA_MANIPULATOR_H

#include <visionaray/detail/macros.h>
#include <visionaray/math/math.h>

#include "../input/mouse.h"


namespace visionaray
{

class camera;
class key_event;
class mouse_event;

class camera_manipulator
{
public:

    camera_manipulator(camera& cam) : camera_(cam) {}
    virtual ~camera_manipulator() {}

    virtual void handle_key_press(key_event const& event)     { VSNRAY_UNUSED(event); }
    virtual void handle_key_release(key_event const& event)   { VSNRAY_UNUSED(event); }

    virtual void handle_mouse_down(mouse_event const& event)  { VSNRAY_UNUSED(event); }
    virtual void handle_mouse_up(mouse_event const& event)    { VSNRAY_UNUSED(event); }
    virtual void handle_mouse_click(mouse_event const& event) { VSNRAY_UNUSED(event); }
    virtual void handle_mouse_move(mouse_event const& event)  { VSNRAY_UNUSED(event); }

public://rotected:

    camera& camera_;

};


class fp_manipulator : public camera_manipulator
{
public:

    fp_manipulator(camera& cam);
   ~fp_manipulator();

    void handle_key_press(key_event const& event);

};

} // visionaray

#endif // VSNRAY_CAMERA_MANIPULATOR_H
