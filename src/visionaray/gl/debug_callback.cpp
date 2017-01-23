// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <iostream>
#include <ostream>
#include <stdexcept>

#include <visionaray/detail/platform.h>

#ifdef VSNRAY_OS_WIN32
#include <windows.h>
#endif

#include <visionaray/config.h>

#if VSNRAY_HAVE_GLEW
#include <GL/glew.h>
#endif

#include <visionaray/gl/debug_callback.h>

#include "../util.h"

namespace visionaray
{
namespace gl
{

#if defined(GL_KHR_debug)

//-------------------------------------------------------------------------------------------------
// Helpers
//

static char const* get_debug_type_string(GLenum type)
{
    switch (type)
    {
    case GL_DEBUG_TYPE_ERROR:
        return "error";
    case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
        return "deprecated behavior detected";
    case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
        return "undefined behavior detected";
    case GL_DEBUG_TYPE_PORTABILITY:
        return "portablility warning";
    case GL_DEBUG_TYPE_PERFORMANCE:
        return "performance warning";
    case GL_DEBUG_TYPE_OTHER:
        return "other";
    case GL_DEBUG_TYPE_MARKER:
        return "marker";
    }

    return "{unknown type}";
}


//-------------------------------------------------------------------------------------------------
// The actual callback function
//

static void debug_callback_func(
        GLenum          /*source*/,
        GLenum          type,
        GLuint          /*id*/,
        GLenum          severity,
        GLsizei         /*length*/,
        const GLchar*   message,
        GLvoid*         user_param
        )
{
    debug_params params = *static_cast<debug_params*>(user_param);

    if (
        // severity
        ( severity == GL_DEBUG_SEVERITY_NOTIFICATION    && params.level <= debug_level::Notification       ) ||
        ( severity == GL_DEBUG_SEVERITY_LOW             && params.level <= debug_level::Low                ) ||
        ( severity == GL_DEBUG_SEVERITY_MEDIUM          && params.level <= debug_level::Medium             ) ||
        ( severity == GL_DEBUG_SEVERITY_HIGH            && params.level <= debug_level::High               ) ||

        // whitelisted message types, override level param
        ( type     == GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR && (params.types & debug_type::DeprecatedBehavior) ) ||
        ( type     == GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR  && (params.types & debug_type::UndefinedBehavior)  ) ||
        ( type     == GL_DEBUG_TYPE_PORTABILITY         && (params.types & debug_type::Portability)        ) ||
        ( type     == GL_DEBUG_TYPE_PERFORMANCE         && (params.types & debug_type::Performance)        ) ||
        ( type     == GL_DEBUG_TYPE_OTHER               && (params.types & debug_type::Other)              )
        )
    {
        std::cerr << "GL " << get_debug_type_string(type) << " " << message << '\n';
    }

    if (type == GL_DEBUG_TYPE_ERROR)
    {
#ifdef _WIN32
        if (IsDebuggerPresent())
        {
            DebugBreak();
        }
#else
        std::cerr << visionaray::util::backtrace() << '\n';
        throw std::runtime_error("OpenGL error");
#endif
    }
}

#endif // GL_KHR_debug


//-------------------------------------------------------------------------------------------------
// Implementation
//

bool debug_callback::activate(debug_params params)
{
    params_ = params;

#if defined(GL_KHR_debug)
    if (GLEW_KHR_debug)
    {
        glEnable(GL_DEBUG_OUTPUT);
        glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);

        glDebugMessageCallback((GLDEBUGPROC)debug_callback_func, (GLvoid*)&params_);

        return true;
    }
#elif defined(GL_ARB_debug_output)
    if (GLEW_ARB_debug_output)
    {
        return false; // TODO
    }
#endif

    return false;
}

} // gl
} // visionaray
