// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_LIGHT_SAMPLE_H
#define VSNRAY_LIGHT_SAMPLE_H 1

namespace visionaray
{

template <typename T>
struct light_sample
{
    vector<3, T> pos;
    vector<3, T> intensity;
    vector<3, T> normal;
    T area;
};

} // visionaray

#endif // VSNRAY_LIGHT_SAMPLE_H
