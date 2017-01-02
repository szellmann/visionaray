// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_GL_TYPES_H
#define VSNRAY_GL_TYPES_H 1

#include <cstddef>

// OpenGL 1.0
typedef unsigned int        GLenum;
typedef unsigned char       GLboolean;
typedef unsigned int        GLbitfield;
typedef void                GLvoid;
typedef signed char         GLbyte;
typedef short               GLshort;
typedef int                 GLint;
typedef unsigned char       GLubyte;
typedef unsigned short      GLushort;
typedef unsigned int        GLuint;
typedef int                 GLsizei;
typedef float               GLfloat;
typedef float               GLclampf;
typedef double              GLdouble;
typedef double              GLclampd;

// OpenGL 1.5
typedef ptrdiff_t           GLintptr;
typedef ptrdiff_t           GLsizeiptr;

// OpenGL 2.0
typedef char                GLchar;

#if 0
// OpenGL 3.0
typedef int64_t             GLint64;
typedef uint64_t            GLuint64;
#endif

#endif // VSNRAY_GL_TYPES_H
