// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_MANIP_MODEL_MANIPULATOR_H
#define VSNRAY_COMMON_MANIP_MODEL_MANIPULATOR_H 1

#include <visionaray/math/forward.h>
#include <visionaray/math/matrix.h>
#include <visionaray/math/vector.h>

namespace visionaray
{

class pinhole_camera;
class key_event;
class mouse_event;

class model_manipulator
{
public:

    model_manipulator(
            pinhole_camera const& cam,
            mat4& model_matrix_,
            vec3 size
            );
    virtual ~model_manipulator();

    void set_active(bool active);
    bool active() const;

    virtual void render();

    virtual bool handle_key_press(key_event const& event);
    virtual bool handle_key_release(key_event const& event);

    virtual bool handle_mouse_down(mouse_event const& event);
    virtual bool handle_mouse_up(mouse_event const& event);
    virtual bool handle_mouse_click(mouse_event const& event);
    virtual bool handle_mouse_move(mouse_event const& event);

protected:

    pinhole_camera const& camera_;
    mat4& model_matrix_;
    vec3 size_;

    bool active_;

};

} // visionaray

#endif // VSNRAY_COMMON_MANIP_MODEL_MANIPULATOR_H
