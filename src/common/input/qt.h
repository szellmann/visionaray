// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_INPUT_QT_H
#define VSNRAY_COMMON_INPUT_QT_H 1

#include <common/config.h>

#if VSNRAY_HAVE_QT5CORE

#include <Qt>

#include <visionaray/detail/platform.h>

#include "keyboard.h"
#include "mouse.h"


namespace visionaray
{
namespace mouse
{

//-------------------------------------------------------------------------------------------------
// Mouse buttons
//

static inline buttons map_qt_button(Qt::MouseButton but)
{
    // TODO: multiple buttons
    switch (but)
    {

    case Qt::LeftButton:
        return mouse::Left;
    case Qt::MiddleButton:
        return mouse::Middle;
    case Qt::RightButton:
        return mouse::Right;
    default:
        return NoButton;

    }

    return NoButton;
}

} // mouse


namespace keyboard
{

//-------------------------------------------------------------------------------------------------
// Keys
//

static inline key map_qt_key(int code, Qt::KeyboardModifiers modifiers = Qt::NoModifier)
{
    bool shift = modifiers & Qt::ShiftModifier;

    switch (code)
    {

    case Qt::Key_A:         return shift ? A : a;
    case Qt::Key_B:         return shift ? B : b;
    case Qt::Key_C:         return shift ? C : c;
    case Qt::Key_D:         return shift ? D : d;
    case Qt::Key_E:         return shift ? E : e;
    case Qt::Key_F:         return shift ? F : f;
    case Qt::Key_G:         return shift ? G : g;
    case Qt::Key_H:         return shift ? H : h;
    case Qt::Key_I:         return shift ? I : i;
    case Qt::Key_J:         return shift ? J : j;
    case Qt::Key_K:         return shift ? K : k;
    case Qt::Key_L:         return shift ? L : l;
    case Qt::Key_M:         return shift ? M : m;
    case Qt::Key_N:         return shift ? N : n;
    case Qt::Key_O:         return shift ? O : o;
    case Qt::Key_P:         return shift ? P : p;
    case Qt::Key_Q:         return shift ? Q : q;
    case Qt::Key_R:         return shift ? R : r;
    case Qt::Key_S:         return shift ? S : s;
    case Qt::Key_T:         return shift ? T : t;
    case Qt::Key_U:         return shift ? U : u;
    case Qt::Key_V:         return shift ? V : v;
    case Qt::Key_W:         return shift ? W : w;
    case Qt::Key_X:         return shift ? X : x;
    case Qt::Key_Y:         return shift ? Y : y;
    case Qt::Key_Z:         return shift ? Z : z;

    case Qt::Key_0:         return Zero;
    case Qt::Key_1:         return One;
    case Qt::Key_2:         return Two;
    case Qt::Key_3:         return Three;
    case Qt::Key_4:         return Four;
    case Qt::Key_5:         return Five;
    case Qt::Key_6:         return Six;
    case Qt::Key_7:         return Seven;
    case Qt::Key_8:         return Eight;
    case Qt::Key_9:         return Nine;

    case Qt::Key_Plus:      return Plus;
    case Qt::Key_Comma:     return Comma;
    case Qt::Key_Minus:     return Minus;
    case Qt::Key_Period:    return Period;

    case Qt::Key_Space:     return Space;
    case Qt::Key_Escape:    return Escape;
    case Qt::Key_Enter:     return Enter;
    case Qt::Key_Tab:       return Tab;

    case Qt::Key_Left:      return ArrowLeft;
    case Qt::Key_Right:     return ArrowRight;
    case Qt::Key_Up:        return ArrowUp;
    case Qt::Key_Down:      return ArrowDown;

    case Qt::Key_PageUp:    return PageUp;
    case Qt::Key_PageDown:  return PageDown;
    case Qt::Key_Home:      return Home;
    case Qt::Key_End:       return End;
    case Qt::Key_Insert:    return Insert;

    case Qt::Key_F1:        return F1;
    case Qt::Key_F2:        return F2;
    case Qt::Key_F3:        return F3;
    case Qt::Key_F4:        return F4;
    case Qt::Key_F5:        return F5;
    case Qt::Key_F6:        return F6;
    case Qt::Key_F7:        return F7;
    case Qt::Key_F8:        return F8;
    case Qt::Key_F9:        return F9;
    case Qt::Key_F10:       return F10;
    case Qt::Key_F11:       return F11;
    case Qt::Key_F12:       return F12;

    default:                return NoKey;

    }

    return NoKey;
}


//-------------------------------------------------------------------------------------------------
// Modifiers
//

static inline key_modifiers map_qt_modifiers(Qt::KeyboardModifiers code)
{
    key_modifiers result = NoKey;

#if defined(VSNRAY_OS_DARWIN)
    if (code & Qt::MetaModifier)
    {
        result |= Ctrl;
    }
#else
    if (code & Qt::ControlModifier)
    {
        result |= Ctrl;
    }
#endif

    if (code & Qt::AltModifier)
    {
        result |= Alt;
    }

    if (code & Qt::ShiftModifier)
    {
        result |= Shift;
    }

    return result;
}

} // keyboard
} // visionaray

#endif // VSNRAY_HAVE_QT5CORE

#endif // VSNRAY_COMMON_INPUT_QT_H
