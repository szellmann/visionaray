// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_THIN_LENS_CAMERA_H
#define VSNRAY_THIN_LENS_CAMERA_H 1

#include "pinhole_camera.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Thin lens camera class
//
//-------------------------------------------------------------------------------------------------

class thin_lens_camera : public pinhole_camera
{
public:

    void set_lens_radius(float lens_radius) { lens_radius_ = lens_radius; }
    float get_lens_radius() const { return lens_radius_; }

    void set_focal_distance(float focal_distance) { focal_distance_ = focal_distance; }
    float get_focal_distance() const { return focal_distance_; }


    // Generate primary ray at (x,y) (may be a subpixel position).
    template <typename R, typename Generator, typename T = typename R::scalar_type>
    VSNRAY_FUNC
    R primary_ray(R /* */, Generator& gen, T const& x, T const& y, T const& width, T const& height) const;

private:

    float lens_radius_;
    float focal_distance_;

};

} // visionaray

#include "detail/thin_lens_camera.inl"

#endif // VSNRAY_THIN_LENS_CAMERA_H
