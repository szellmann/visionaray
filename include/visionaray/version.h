// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_VERSION_H
#define VSNRAY_VERSION_H 1

#define VSNRAY_VERSION_MAJOR 0
#define VSNRAY_VERSION_MINOR 1
#define VSNRAY_VERSION_PATCH 1


//-------------------------------------------------------------------------------------------------
// Convenience macros to obtain version info
// VSNRAY_VERSION_{EQ|NEQ|LT|LE|GT|GE} (==|!=|<|<=|>|>=)
//
// Use like:
//
// #if VSNRAY_VERSION_GE(0,0,1)
//     // Use features that require Visionaray version 0.0.1 or newer
// #endif
//

#define VSNRAY_VERSION_EQ(MAJOR, MINOR, PATCH)                      \
       (VSNRAY_VERSION_MAJOR == MAJOR)                              \
    && (VSNRAY_VERSION_MINOR == MINOR)                              \
    && (VSNRAY_VERSION_PATCH == PATCH)

#define VSNRAY_VERSION_NEQ(MAJOR, MINOR, PATCH)                     \
    !( VSNRAY_VERSION_EQ(MAJOR, MINOR, PATCH) )

#define VSNRAY_VERSION_LT(MAJOR, MINOR, PATCH)                      \
       (VSNRAY_VERSION_MAJOR < MAJOR)                               \
    || ( (VSNRAY_VERSION_MAJOR == MAJOR)                            \
      && (VSNRAY_VERSION_MINOR < MINOR) )                           \
    || ( (VSNRAY_VERSION_MAJOR == MAJOR)                            \
      && (VSNRAY_VERSION_MINOR == MINOR)                            \
      && (VSNRAY_VERSION_PATCH < PATCH) )

#define VSNRAY_VERSION_LE(MAJOR, MINOR, PATCH)                      \
       ( VSNRAY_VERSION_LT(MAJOR, MINOR, PATCH) )                   \
    || ( VSNRAY_VERSION_EQ(MAJOR, MINOR, PATCH) )

#define VSNRAY_VERSION_GT(MAJOR, MINOR, PATCH)                      \
    !( VSNRAY_VERSION_LE(MAJOR, MINOR, PATCH) )

#define VSNRAY_VERSION_GE(MAJOR, MINOR, PATCH)                      \
    !( VSNRAY_VERSION_LT(MAJOR, MINOR, PATCH) )

#endif // VSNRAY_VERSION_H
