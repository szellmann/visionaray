// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_INPUT_KEYBOARD_H
#define VSNRAY_COMMON_INPUT_KEYBOARD_H 1

namespace visionaray
{
namespace keyboard
{

enum key
{
    A = 0x41, B = 0x42, C = 0x43, D = 0x44, E = 0x45,
    F = 0x46, G = 0x47, H = 0x48, I = 0x49, J = 0x4A,
    K = 0x4B, L = 0x4C, M = 0x4D, N = 0x4E, O = 0x4F,
    P = 0x50, Q = 0x51, R = 0x52, S = 0x53, T = 0x54,
    U = 0x55, V = 0x56, W = 0x57, X = 0x58, Y = 0x59,
    Z = 0x5A,

    a = 0x61, b = 0x62, c = 0x63, d = 0x64, e = 0x65,
    f = 0x66, g = 0x67, h = 0x68, i = 0x69, j = 0x6A,
    k = 0x6B, l = 0x6C, m = 0x6D, n = 0x6E, o = 0x6F,
    p = 0x70, q = 0x71, r = 0x72, s = 0x73, t = 0x74,
    u = 0x75, v = 0x76, w = 0x77, x = 0x78, y = 0x79,
    z = 0x7A,

    Zero = 0x30, One = 0x31, Two = 0x32, Three = 0x33, Four = 0x34,
    Five = 0x35, Six = 0x36, Seven = 0x37, Eight = 0x38, Nine = 0x39,

    Plus = 0x2B, Comma = 0x2C, Minus = 0x2D, Period = 0x2E,

    ArrowLeft, ArrowRight, ArrowUp, ArrowDown,

    PageUp, PageDown, Home, End, Insert,

    Space = 0x20, Escape = 0x1B, Enter = 0x0D, Tab = 0x9,

    F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12,

    // modifiers, can be combined bitwise

    Ctrl  = 0x00000001,
    Alt   = 0x00000002,
    Shift = 0x00000004,

    NoKey = 0x00000000
};


using key_modifiers = key;


enum event_type
{
    KeyPress = 0,
    KeyRelease
};


//-------------------------------------------------------------------------------------------------
// Bitwise operations on keys
//

inline key_modifiers operator&(key_modifiers a, key b)
{
    return static_cast<key_modifiers>( static_cast<int>(a) & static_cast<int>(b) );
}

inline key_modifiers operator|(key_modifiers a, key b)
{
    return static_cast<key_modifiers>( static_cast<int>(a) | static_cast<int>(b) );
}

inline key_modifiers operator^(key_modifiers a, key b)
{
    return static_cast<key_modifiers>( static_cast<int>(a) ^ static_cast<int>(b) );
}

inline key_modifiers& operator&=(key_modifiers& a, key b)
{
    a = a & b;
    return a;
}

inline key_modifiers& operator|=(key_modifiers& a, key b)
{
    a = a | b;
    return a;
}

inline key_modifiers& operator^=(key_modifiers& a, key b)
{
    a = a ^ b;
    return a;
}

} // keyboard
} // visionaray

#endif // VSNRAY_COMMON_INPUT_KEYBOARD_H
