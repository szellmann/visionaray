// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_INPUT_GLUT_H
#define VSNRAY_COMMON_INPUT_GLUT_H 1

#include <common/config.h>

#if VSNRAY_COMMON_HAVE_GLUT

#include <visionaray/detail/platform.h>

#if defined(VSNRAY_OS_DARWIN)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include "keyboard.h"
#include "mouse.h"


namespace visionaray
{

namespace mouse
{

//-------------------------------------------------------------------------------------------------
// Mouse buttons
//

inline buttons map_glut_button(int but)
{
    // GLUT callbacks don't handle multiple buttons pressed at once
    switch (but)
    {

    case GLUT_LEFT_BUTTON:
        return mouse::Left;
    case GLUT_MIDDLE_BUTTON:
        return mouse::Middle;
    case GLUT_RIGHT_BUTTON:
        return mouse::Right;

    }

    return NoButton;
}

} // mouse


namespace keyboard
{

//-------------------------------------------------------------------------------------------------
// Keys
//

inline key map_glut_key(unsigned char code)
{

    switch (code)
    {

    case 'A':                   return A;
    case 'B':                   return B;
    case 'C':                   return C;
    case 'D':                   return D;
    case 'E':                   return E;
    case 'F':                   return F;
    case 'G':                   return G;
    case 'H':                   return H;
    case 'I':                   return I;
    case 'J':                   return J;
    case 'K':                   return K;
    case 'L':                   return L;
    case 'M':                   return M;
    case 'N':                   return N;
    case 'O':                   return O;
    case 'P':                   return P;
    case 'Q':                   return Q;
    case 'R':                   return R;
    case 'S':                   return S;
    case 'T':                   return T;
    case 'U':                   return U;
    case 'V':                   return V;
    case 'W':                   return W;
    case 'X':                   return X;
    case 'Y':                   return Y;
    case 'Z':                   return Z;

    case 'a':                   return a;
    case 'b':                   return b;
    case 'c':                   return c;
    case 'd':                   return d;
    case 'e':                   return e;
    case 'f':                   return f;
    case 'g':                   return g;
    case 'h':                   return h;
    case 'i':                   return i;
    case 'j':                   return j;
    case 'k':                   return k;
    case 'l':                   return l;
    case 'm':                   return m;
    case 'n':                   return n;
    case 'o':                   return o;
    case 'p':                   return p;
    case 'q':                   return q;
    case 'r':                   return r;
    case 's':                   return s;
    case 't':                   return t;
    case 'u':                   return u;
    case 'v':                   return v;
    case 'w':                   return w;
    case 'x':                   return x;
    case 'y':                   return y;
    case 'z':                   return z;


    case '0':                   return Zero;
    case '1':                   return One;
    case '2':                   return Two;
    case '3':                   return Three;
    case '4':                   return Four;
    case '5':                   return Five;
    case '6':                   return Six;
    case '7':                   return Seven;
    case '8':                   return Eight;
    case '9':                   return Nine;

    case '+':                   return Plus;
    case ',':                   return Comma;
    case '-':                   return Minus;
    case '.':                   return Period;

    case 0x20:                  return Space;
    case 0x1B:                  return Escape;
    case 0x0D:                  return Enter;
    case 0x9:                   return Tab;

    }

    return NoKey;
}

inline key map_glut_special(int code)
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


//-------------------------------------------------------------------------------------------------
// Modifiers
//

inline key_modifiers map_glut_modifiers(unsigned char code)
{
    key_modifiers result = NoKey;

    if (code & GLUT_ACTIVE_CTRL)
    {
        result |= Ctrl;
    }

    if (code & GLUT_ACTIVE_ALT)
    {
        result |= Alt;
    }

    if (code & GLUT_ACTIVE_SHIFT)
    {
        result |= Shift;
    }

    return result;
}

} // keyboard
} // visionaray

#endif // VSNRAY_COMMON_HAVE_GLUT

#endif // VSNRAY_COMMON_INPUT_GLUT_H
