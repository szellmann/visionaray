// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VNSRAY_INPUT_MOUSE_H
#define VSNRAY_INPUT_MOUSE_H

#include <bitset>
#include <cassert>

#include <visionaray/detail/platform.h>

#if defined(VSNRAY_OS_DARWIN)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

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

typedef button buttons;//std::bitset<4> buttons;


enum event_type
{
    ButtonClick = 0,
    ButtonDblClick,
    ButtonDown,
    ButtonMove,
    ButtonUp,
    Move
};


struct pos
{
    int x;
    int y;
};


static inline button map_glut_button(int but)
{
    switch (but)
    {

    case GLUT_LEFT_BUTTON:      return mouse::Left;
    case GLUT_MIDDLE_BUTTON:    return mouse::Middle;
    case GLUT_RIGHT_BUTTON:     return mouse::Right;

    }

    return NoButton;
}


} // mouse


class mouse_event
{
public:

    mouse_event(mouse::event_type type, mouse::button button, mouse::pos const& pos, mouse::buttons buttons = mouse::NoButton)
        : type_(type)
        , button_(button)
        , pos_(pos)
        , buttons_(buttons)
        , modifiers_(keyboard::NoKey)
    {
    }

    mouse_event(mouse::event_type type, mouse::button button, mouse::pos const& pos, mouse::buttons buttons, keyboard::key_modifiers modifiers)
        : type_(type)
        , button_(button)
        , pos_(pos)
        , buttons_(buttons)
        , modifiers_(modifiers)
    {

        using namespace keyboard;

        key_modifiers tmp = modifiers;

        // TODO: why does implicit bitset c'tor (unsigned long) not apply?
/*        tmp ^= ( tmp & static_cast<key_modifiers>(Alt) );
        tmp ^= ( tmp & static_cast<key_modifiers>(Ctrl) );
        tmp ^= ( tmp & static_cast<key_modifiers>(Shift) );*/

        if (tmp != NoKey)
        {
            throw invalid_key_modifier();
        }

    }

    mouse::event_type get_type() const { return type_; }
    mouse::button get_button()   const { return button_; }
    mouse::pos const& get_pos()  const { return pos_; }
    mouse::buttons get_buttons() const { return buttons_; }
    keyboard::key_modifiers get_modifiers() const { return modifiers_; }

private:

    mouse::event_type       type_;

    //! Button that caused the event
    mouse::button           button_;

    mouse::pos              pos_;

    //! Button state during the event
    mouse::buttons          buttons_;

    keyboard::key_modifiers modifiers_;

};

} // visionaray

#endif // VSNRAY_INPUT_MOUSE_H
