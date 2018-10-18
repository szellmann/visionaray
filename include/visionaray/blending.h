// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_BLENDING_H
#define VSNRAY_BLENDING_H 1

namespace visionaray
{
namespace blending
{

enum scale_factor
{
    Zero,
    One,
    SrcColor,
    OneMinusSrcColor,
    DstColor,
    OneMinusDstColor,
    SrcAlpha,
    OneMinusSrcAlpha,
    DstAlpha,
    OneMinusDstAlpha,
};

} // blending
} // visionaray

#endif // VSNRAY_BLENDING_H
