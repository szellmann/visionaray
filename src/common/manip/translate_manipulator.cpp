// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

#include <GL/glew.h>

#include <visionaray/math/math.h>
#include <visionaray/pinhole_camera.h>

#include "../input/mouse.h"
#include "../input/mouse_event.h"
#include "translate_manipulator.h"

using namespace visionaray;


//-------------------------------------------------------------------------------------------------
// Projection
//

static vec2i get_projected_center(pinhole_camera const& cam, mat4 const& model_matrix)
{
    vec3 win;
    vec3 obj(0.0f);

    project(
        win,
        obj,
        cam.get_view_matrix() * model_matrix,
        cam.get_proj_matrix(),
        cam.get_viewport()
        );

    return vec2i(win.xy());
}

static recti get_projected_center_rect(pinhole_camera const& cam, mat4 const& model_matrix)
{
    vec2i c = get_projected_center(cam, model_matrix);
    return recti(c.x - 10, c.y - 10, 20, 20);
}


//-------------------------------------------------------------------------------------------------
// Convert from Visionaray / GL window coordinates to left/upper origin coordinates
// TODO: copied from rotate_manipulator - consolidate
//

static recti flip(recti const& r, recti const& viewport)
{
    recti result = r;
    result.y = viewport.h - r.h - r.y - 1;
    return result;
}


//-------------------------------------------------------------------------------------------------
// Draw a horizontal or vertical line in screen space
// TODO: don't use fixed-function pipeline
//

void draw_line(vec2i offset, int len, vec4 color, cartesian_axis<2> axis, int thickness = 1)
{
    int num_pixels = len * thickness;

    using C = vector<4, unorm<8>>;

    std::vector<C> pixels(num_pixels);
    std::fill(pixels.begin(), pixels.end(), C(color));

    int w = axis == cartesian_axis<2>::X ? len : thickness;
    int h = axis == cartesian_axis<2>::X ? thickness : len;

    glWindowPos2i(offset.x, offset.y);
    glDrawPixels(w, h, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
}


//-------------------------------------------------------------------------------------------------
//
//

translate_manipulator::translate_manipulator(
        pinhole_camera const& cam,
        mat4& model_matrix,
        vec3 size,
        mouse::buttons buttons
        )
    : model_manipulator(cam, model_matrix, size)
    , buttons_(buttons)
{
}

void translate_manipulator::render()
{
    glPushAttrib(GL_CURRENT_BIT | GL_TRANSFORM_BIT);

    auto size = max_element(get_scaling(model_matrix_)/* really? */ * vec4(size_, 1.0f));

    vec2i c = get_projected_center(camera_, model_matrix_);

    vec4 color(0.0f, 1.0f, 1.0f, 1.0f);
    draw_line(c + vec2i(-10, -10), 20, color, cartesian_axis<2>::X, 1);
    draw_line(c + vec2i( 10, -10), 20, color, cartesian_axis<2>::Y, 1);
    draw_line(c + vec2i(-10,  10), 20, color, cartesian_axis<2>::X, 1);
    draw_line(c + vec2i(-10, -10), 20, color, cartesian_axis<2>::Y, 1);


    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadMatrixf(camera_.get_proj_matrix().data());

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadMatrixf(camera_.get_view_matrix().data());
    glMultMatrixf(model_matrix_.data());

    glBegin(GL_LINES);
        glColor3f(1.0f, 0.0f, 0.0f);
        glVertex3f(0.2f, 0.0f, 0.0f);
        glVertex3f(size, 0.0f, 0.0f);

        glColor3f(0.0f, 1.0f, 0.0f);
        glVertex3f(0.0f, 0.2f, 0.0f);
        glVertex3f(0.0f, size, 0.0f);

        glColor3f(0.0f, 0.0f, 1.0f);
        glVertex3f(0.0f, 0.0f, 0.2f);
        glVertex3f(0.0f, 0.0f, size);
    glEnd();

    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    glPopAttrib();
}

bool translate_manipulator::handle_mouse_down(visionaray::mouse_event const& event)
{
    auto brect = flip(
            get_projected_center_rect(camera_, model_matrix_),
            camera_.get_viewport()
            );

    if (brect.contains(event.get_pos()) && event.get_buttons() & buttons_)
    {
        dragging_ = true;
        down_pos_ = event.get_pos();

        return true;
    }
    else
    {
        return false;
    }
}

bool translate_manipulator::handle_mouse_up(visionaray::mouse_event const& event)
{
    VSNRAY_UNUSED(event);

    dragging_ = false;
    return false;
}

bool translate_manipulator::handle_mouse_move(visionaray::mouse_event const& event)
{
    if (dragging_)
    {
        auto last_pos_ = down_pos_;

        auto w  =  camera_.get_viewport().w;
        auto h  =  camera_.get_viewport().h;
        auto dx = -static_cast<float>(last_pos_.x - event.get_pos().x) / w;
        auto dy = +static_cast<float>(last_pos_.y - event.get_pos().y) / h;
        auto s  = 2.0f * camera_.distance();
        auto Z = normalize( camera_.eye() - camera_.center() );
        auto Y = camera_.up();
        auto X = cross(Y, Z);
        vec3 d  =  (dx * s) * X + (dy * s) * Y;

        model_matrix_ = mat4::translation(d) * model_matrix_;

        down_pos_ = event.get_pos();

        return true;
    }
    else
    {
        return false;
    }
}
