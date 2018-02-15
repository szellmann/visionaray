// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_SURFACE_INTERACTION_H
#define VSNRAY_SURFACE_INTERACTION_H 1

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Interaction type to track the surface interaction that occurred
//

struct surface_interaction
{
    static const int Unspecified          = 1 << 0;
    static const int Emission             = 1 << 1;
    static const int Diffuse              = 1 << 2;
    static const int SpecularReflection   = 1 << 3;
    static const int SpecularTransmission = 1 << 4;
    static const int GlossyReflection     = 1 << 5;
    static const int GlossyTransmission   = 1 << 6;
};

} // visionaray

#endif // VSNRAY_SURFACE_INTERACTION_H
