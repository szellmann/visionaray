// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_INPUT_MOUSE_H
#define VSNRAY_COMMON_INPUT_MOUSE_H 1

#include <visionaray/math/vector.h>

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

} // visionaray

#endif // VSNRAY_COMMON_INPUT_MOUSE_H
