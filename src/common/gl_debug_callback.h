// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_GL_DEBUG_CALLBACK_H
#define VSNRAY_COMMON_GL_DEBUG_CALLBACK_H 1

namespace visionaray
{
namespace gl
{

//-------------------------------------------------------------------------------------------------
// Wrapper class for GL_KHR_debug
//
//
// Store an instance of class debug_callback at the scope
// of the thread that created the OpenGL context
//
// Call activate() to enable logging
// Optionally pass params to control log level to activate()
//
//
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
// Flags used in debug callback function
//

enum debug_level
{
    Notification,
    Low,
    Medium,
    High
};

enum debug_type
{
    // can be combined bitwise

    Error               = 0x00000001,
    DeprecatedBehavior  = 0x00000002,
    UndefinedBehavior   = 0x00000004,
    Portability         = 0x00000008,
    Performance         = 0x00000010,
    Other               = 0x00000020,

    None                = 0x00000000
};


//-------------------------------------------------------------------------------------------------
// Debug parameters passed to debug callback function
//
//  param level
//      filter messages by severity
//
//  param types
//      overrides param level
//      whitelist several debug message types
//      bitwise combination of debug_type values
//

struct debug_params
{
    debug_level level   = debug_level::High;
    debug_type  types   = debug_type::None;
};


//-------------------------------------------------------------------------------------------------
// Debug callback class
//

class debug_callback
{
public:

    bool activate(debug_params params = debug_params());

private:

    debug_params params_;

};

} // gl
} // visionaray

#endif // VSNRAY_COMMON_GL_DEBUG_CALLBACK_H
