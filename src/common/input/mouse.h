// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_INPUT_MOUSE_H
#define VSNRAY_COMMON_INPUT_MOUSE_H 1

#include <visionaray/math/vector.h>

#include "exception.h"
#include "keyboard.h"

namespace visionaray
{
namespace mouse
{


enum button
{
    Left     = 0x1,
    Middle   = 0x2,
    Right    = 0x4,

    NoButton = 0x0
};

using buttons = button;

enum event_type
{
    ButtonClick = 0,
    ButtonDblClick,
    ButtonDown,
    ButtonMove,
    ButtonUp,
    Move
};

using pos = vector<2, int>;


//-------------------------------------------------------------------------------------------------
// Bitwise operations on buttons
//

inline buttons operator&(buttons a, button b)
{
    return static_cast<buttons>( static_cast<int>(a) & static_cast<int>(b) );
}

inline buttons operator|(buttons a, button b)
{
    return static_cast<buttons>( static_cast<int>(a) | static_cast<int>(b) );
}

inline buttons operator^(buttons a, button b)
{
    return static_cast<buttons>( static_cast<int>(a) ^ static_cast<int>(b) );
}

inline buttons& operator&=(buttons& a, button b)
{
    a = a & b;
    return a;
}

inline buttons& operator|=(buttons& a, button b)
{
    a = a | b;
    return a;
}

inline buttons& operator^=(buttons& a, button b)
{
    a = a ^ b;
    return a;
}

} // mouse


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

    mouse::event_type get_type() const { return type_; }
    mouse::pos const& get_pos()  const { return pos_; }
    mouse::buttons get_buttons() const { return buttons_; }
    keyboard::key_modifiers get_modifiers() const { return modifiers_; }

private:

    mouse::event_type       type_;

    mouse::pos              pos_;

    //! Button state during the event
    mouse::buttons          buttons_;

    keyboard::key_modifiers modifiers_;

};

} // visionaray

#endif // VSNRAY_COMMON_INPUT_MOUSE_H
