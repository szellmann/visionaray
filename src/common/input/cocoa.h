// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_INPUT_COCOA_H
#define VSNRAY_COMMON_INPUT_COCOA_H 1

#include <Cocoa/Cocoa.h>

#include "keyboard.h"
#include "mouse.h"


namespace visionaray
{

namespace mouse
{

//-------------------------------------------------------------------------------------------------
// Mouse buttons
//

inline buttons map_cocoa_button(NSEventType et)
{
    switch (et)
    {

    case NSEventTypeLeftMouseDown:
    case NSEventTypeLeftMouseDragged:
    case NSEventTypeLeftMouseUp:
        return mouse::Left;

//  case NSEventTypeOtherMouseDown:
//  case NSEventTypeOtherMouseDragged:
//  case NSEventTypeOtherMouseUp:
        // fall-through
    case NSEventTypeRightMouseDown:
    case NSEventTypeRightMouseDragged:
    case NSEventTypeRightMouseUp:
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

inline key map_cocoa_key(unsigned char code)
{
    if (code >= 0 && code < 128) // ascii
    {
        return static_cast<key>(code);
    }

    return NoKey;
}


//-----------------------------------------------------------------------------    --------------------
// Modifiers
//

inline key_modifiers map_cocoa_modifiers(NSUInteger code)
{
    key_modifiers result = NoKey;

    if (code & NSEventModifierFlagOption)
    {
        result |= Alt;
    }

    if (code & NSEventModifierFlagControl)
    {
        result |= Ctrl;
    }

    if (code & NSEventModifierFlagShift)
    {
        result |= Shift;
    }

    return result;
}

} // keyboard

} // visionaray

#endif // VSNRAY_COMMON_INPUT_COCOA_H
