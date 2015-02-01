// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_INPUT_KEYBOARD_H
#define VSNRAY_INPUT_KEYBOARD_H

#include <cassert>
#include <bitset>

#include <visionaray/detail/platform.h>

#if defined(VSNRAY_OS_DARWIN)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif


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

    Zero = 0x30, One = 0x31, Two = 0x32, Three = 0x33, Four = 0x34,
    Five = 0x35, Six = 0x36, Seven = 0x37, Eight = 0x38, Nine = 0x39,

    ArrowLeft, ArrowRight, ArrowUp, ArrowDown,

    PageUp, PageDown, Home, End, Insert,

    Space, Escape, Enter, Tab,

    F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12,

    // modifiers, can be combined bitwise

    Ctrl  = 0x00000001,
    Alt   = 0x00000002,
    Shift = 0x00000004,

    NoKey = 0x00000000
};


typedef std::bitset<4> key_modifiers;


static inline key map_glut_key(unsigned char code)
{

    switch (code)
    {

    case 'a':                   return A;
    case 'b':                   return B;
    case 'c':                   return C;
    case 'd':                   return D;
    case 'e':                   return E;
    case 'f':                   return F;
    case 'g':                   return G;
    case 'h':                   return H;
    case 'i':                   return I;
    case 'j':                   return J;
    case 'k':                   return K;
    case 'l':                   return L;
    case 'm':                   return M;
    case 'n':                   return N;
    case 'o':                   return O;
    case 'p':                   return P;
    case 'q':                   return Q;
    case 'r':                   return R;
    case 's':                   return S;
    case 't':                   return T;
    case 'u':                   return U;
    case 'v':                   return V;
    case 'w':                   return W;
    case 'x':                   return X;
    case 'y':                   return Y;
    case 'z':                   return Z;

    case '1':                   return One;
    case '2':                   return Two;
    case '3':                   return Three;
    case '4':                   return Four;
    case '5':                   return Five;
    case '6':                   return Six;
    case '7':                   return Seven;
    case '8':                   return Eight;
    case '9':                   return Nine;

    }

    return NoKey;
}


static inline key map_glut_special(unsigned char code)
{
    switch (code)
    {

    case GLUT_KEY_LEFT:         return ArrowLeft;
    case GLUT_KEY_RIGHT:        return ArrowRight;
    case GLUT_KEY_UP:           return ArrowUp;
    case GLUT_KEY_DOWN:         return ArrowDown;

    case GLUT_KEY_PAGE_UP:      return PageUp;
    case GLUT_KEY_PAGE_DOWN:    return PageDown;
    case GLUT_KEY_HOME:         return Home;
    case GLUT_KEY_END:          return End;
    case GLUT_KEY_INSERT:       return Insert;

    case GLUT_KEY_F1:           return F1;
    case GLUT_KEY_F2:           return F2;
    case GLUT_KEY_F3:           return F3;
    case GLUT_KEY_F4:           return F4;
    case GLUT_KEY_F5:           return F5;
    case GLUT_KEY_F6:           return F6;
    case GLUT_KEY_F7:           return F7;
    case GLUT_KEY_F8:           return F8;
    case GLUT_KEY_F9:           return F9;
    case GLUT_KEY_F10:          return F10;
    case GLUT_KEY_F11:          return F11;
    case GLUT_KEY_F12:          return F12;

    }


    return NoKey;
}


static inline key map_glut_modifier(unsigned char code)
{
    switch (code)
    {

    case GLUT_ACTIVE_CTRL:      return Ctrl;
    case GLUT_ACTIVE_ALT:       return Alt;
    case GLUT_ACTIVE_SHIFT:     return Shift;

    }

    return NoKey;
}


} // keyboard


class key_event
{
public:

    key_event(keyboard::key key) : key_(key), modifiers_(keyboard::NoKey) {}
    key_event(keyboard::key key, keyboard::key_modifiers modifiers) : key_(key), modifiers_(modifiers) {}

    keyboard::key key() const { return key_; }
    keyboard::key_modifiers modifiers() const { return modifiers_; }

private:

    keyboard::key key_;
    keyboard::key_modifiers modifiers_;

};


} // visionaray

#endif // VSNRAY_INPUT_KEYBOARD_H


