// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_MANIP_FIRST_PERSON_MANIPULATOR_H
#define VSNRAY_COMMON_MANIP_FIRST_PERSON_MANIPULATOR_H 1

#include <visionaray/math/math.h>

#include "../input/keyboard.h"
#include "../input/mouse.h"
#include "camera_manipulator.h"

#include <unordered_map>

namespace visionaray
{

class pinhole_camera;
class mouse_event;

class first_person_manipulator : public camera_manipulator
{
public:

    enum move_action
    {
        Forward,
        Backward,
        Left,
        Right,
        Up,
        Down,
    };

    struct enum_hash
    {
        template <typename T>
        size_t operator()(T t) const
        {   
            return static_cast<size_t>(t);
        }   
    };

    using action_key_map = std::unordered_map<keyboard::key, first_person_manipulator::move_action, enum_hash>;

public:

    first_person_manipulator(
            pinhole_camera& cam,
            mouse::buttons buttons,
            action_key_map const& action_keys = {
                { keyboard::w, first_person_manipulator::Forward },
                { keyboard::s, first_person_manipulator::Backward },
                { keyboard::a, first_person_manipulator::Left },
                { keyboard::d, first_person_manipulator::Right },
                { keyboard::r, first_person_manipulator::Up },
                { keyboard::f, first_person_manipulator::Down }
                }
            );
   ~first_person_manipulator();

    void handle_key_press(key_event const& event);

    void handle_mouse_down(mouse_event const& event);
    void handle_mouse_up(mouse_event const& event);
    void handle_mouse_move(mouse_event const& event);

private:

    mouse::buttons buttons_;
    keyboard::key_modifiers modifiers_;

    keyboard::key_modifiers down_modifiers_;

    bool dragging_;

    mouse::pos last_pos_;

    action_key_map action_keys_;

};

} // visionaray

#endif // VSNRAY_COMMON_MANIP_FIRST_PERSON_MANIPULATOR_H
