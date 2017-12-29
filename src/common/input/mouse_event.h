// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_INPUT_MOUSE_EVENT_H
#define VSNRAY_COMMON_INPUT_MOUSE_EVENT_H 1

#include "keyboard.h"
#include "mouse.h"

namespace visionaray
{

class mouse_event
{
public:

    mouse_event(
            mouse::event_type type,
            mouse::pos const& pos,
            mouse::buttons buttons = mouse::NoButton
            )
        : type_(type)
        , pos_(pos)
        , buttons_(buttons)
        , modifiers_(keyboard::NoKey)
    {
    }

    mouse_event(
            mouse::event_type type,
            mouse::pos const& pos,
            mouse::buttons buttons,
            keyboard::key_modifiers modifiers
            )
        : type_(type)
        , pos_(pos)
        , buttons_(buttons)
        , modifiers_(modifiers)
    {
    }

    mouse::event_type type() const { return type_; }
    mouse::pos const& pos()  const { return pos_; }
    mouse::buttons buttons() const { return buttons_; }
    keyboard::key_modifiers modifiers() const { return modifiers_; }

private:

    mouse::event_type       type_;

    mouse::pos              pos_;

    //! Button state during the event
    mouse::buttons          buttons_;

    keyboard::key_modifiers modifiers_;

};

} // visionaray

#endif // VSNRAY_COMMON_INPUT_MOUSE_EVENT_H
