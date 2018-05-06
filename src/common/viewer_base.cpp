// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <common/config.h>

#include <iostream>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#if VSNRAY_COMMON_HAVE_GLEW
#include <GL/glew.h> // glViewport() (TODO!)
#endif

#include <Support/CmdLine.h>
#include <Support/CmdLineUtil.h>

#include "input/key_event.h"
#include "input/keyboard.h"
#include "input/mouse.h"
#include "input/mouse_event.h"
#include "manip/camera_manipulator.h"
#include "viewer_base.h"

using namespace support;
using namespace visionaray;

using manipulators      = std::vector<std::shared_ptr<camera_manipulator>>;
using cmdline_options   = std::vector<std::shared_ptr<cl::OptionBase>>;

struct viewer_base::impl
{
    manipulators    manips;
    cmdline_options options;
    cl::CmdLine     cmd;
    bool            allow_unknown_args = false;

    bool            full_screen        = false;
    int             width              = 512;
    int             height             = 512;
    char const*     window_title       = "";
    vec3            bgcolor            = { 0.1f, 0.4f, 1.0f };

    impl(int width, int height, char const* window_title);

    void init(int argc, char** argv);

    void parse_cmd_line(int argc, char** argv);
};


viewer_base::impl::impl(int width, int height, char const* window_title)
    : width(width)
    , height(height)
    , window_title(window_title)
{
    // add default options (-fullscreen, -width, -height, -bgcolor)

    options.emplace_back( cl::makeOption<bool&>(
        cl::Parser<>(),
        "fullscreen",
        cl::Desc("Full screen window"),
        cl::ArgDisallowed,
        cl::init(viewer_base::impl::full_screen)
        ) );

    options.emplace_back( cl::makeOption<int&>(
        cl::Parser<>(),
        "width",
        cl::Desc("Window width"),
        cl::ArgRequired,
        cl::init(viewer_base::impl::width)
        ) );

    options.emplace_back( cl::makeOption<int&>(
        cl::Parser<>(),
        "height",
        cl::Desc("Window height"),
        cl::ArgRequired,
        cl::init(viewer_base::impl::height)
        ) );

    options.emplace_back( cl::makeOption<vec3&, cl::ScalarType>(
        [&](StringRef name, StringRef /*arg*/, vec3& value)
        {
            cl::Parser<>()(name + "-r", cmd.bump(), value.x);
            cl::Parser<>()(name + "-g", cmd.bump(), value.y);
            cl::Parser<>()(name + "-b", cmd.bump(), value.z);
        },
        "bgcolor",
        cl::Desc("Background color"),
        cl::ArgDisallowed,
        cl::init(viewer_base::impl::bgcolor)
        ) );
}

void viewer_base::impl::init(int argc, char** argv)
{
    try
    {
        parse_cmd_line(argc, argv);
    }
    catch (...)
    {
        std::cout << cmd.help(argv[0]) << '\n';
        throw;
    }
}


//-------------------------------------------------------------------------------------------------
// Parse default command line options
//

void viewer_base::impl::parse_cmd_line(int argc, char** argv)
{
    for (auto& opt : options)
    {
        cmd.add(*opt);
    }

    auto args = std::vector<std::string>(argv + 1, argv + argc);
    cl::expandWildcards(args);
    cl::expandResponseFiles(args, cl::TokenizeUnix());

    cmd.parse(args, allow_unknown_args);
}


viewer_base::viewer_base(
        int width,
        int height,
        char const* window_title
        )
    : impl_(new impl(width, height, window_title))
{
}

viewer_base::~viewer_base()
{
}

void viewer_base::init(int argc, char** argv)
{
    impl_->init(argc, argv);
}

void viewer_base::add_manipulator( std::shared_ptr<camera_manipulator> manip )
{
    impl_->manips.push_back(manip);
}

void viewer_base::add_cmdline_option( std::shared_ptr<cl::OptionBase> option )
{
    impl_->options.emplace_back(option);
}

char const* viewer_base::window_title() const
{
    return impl_->window_title;
}

bool viewer_base::full_screen() const
{
    return impl_->full_screen;
}

int viewer_base::width() const
{
    return impl_->width;
}

int viewer_base::height() const
{
    return impl_->height;
}

vec3 viewer_base::background_color() const
{
    return impl_->bgcolor;
}

void viewer_base::set_allow_unknown_cmd_line_args(bool allow)
{
    impl_->allow_unknown_args = allow;
}

cl::CmdLine& viewer_base::cmd_line_inst()
{
    return impl_->cmd;
}

void viewer_base::event_loop()
{
}

void viewer_base::resize(int width, int height)
{
    impl_->width = width;
    impl_->height = height;
}

void viewer_base::swap_buffers()
{
}

void viewer_base::toggle_full_screen()
{
    impl_->full_screen = !impl_->full_screen;
}

void viewer_base::quit()
{
}


//-------------------------------------------------------------------------------------------------
// Event handlers
//

void viewer_base::on_close()
{
}

void viewer_base::on_display()
{
}

void viewer_base::on_idle()
{
}

void viewer_base::on_key_press(visionaray::key_event const& event)
{
    if (event.key() == keyboard::F5)
    {
        toggle_full_screen();
    }

    if (event.key() == keyboard::Escape && impl_->full_screen)
    {
        toggle_full_screen();
    }

    if (event.key() == keyboard::q)
    {
        quit();
    }

    for (auto& manip : impl_->manips)
    {
        manip->handle_key_press(event);
    }
}

void viewer_base::on_key_release(visionaray::key_event const& event)
{
    for (auto& manip : impl_->manips)
    {
        manip->handle_key_release(event);
    }
}

void viewer_base::on_mouse_move(visionaray::mouse_event const& event)
{
    for (auto& manip : impl_->manips)
    {
        manip->handle_mouse_move(event);
    }
}

void viewer_base::on_mouse_down(visionaray::mouse_event const& event)
{
    for (auto& manip : impl_->manips)
    {
        manip->handle_mouse_down(event);
    }
}

void viewer_base::on_mouse_up(visionaray::mouse_event const& event)
{
    for (auto& manip : impl_->manips)
    {
        manip->handle_mouse_up(event);
    }
}

void viewer_base::on_resize(int w, int h)
{
    impl_->width = w;
    impl_->height = h;

    glViewport(0, 0, w, h);
}
