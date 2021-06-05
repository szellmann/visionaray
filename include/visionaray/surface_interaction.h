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
    enum { Unspecified          = 1 << 0 };
    enum { Emission             = 1 << 1 };
    enum { Diffuse              = 1 << 2 };
    enum { SpecularReflection   = 1 << 3 };
    enum { SpecularTransmission = 1 << 4 };
    enum { GlossyReflection     = 1 << 5 };
    enum { GlossyTransmission   = 1 << 6 };
};

} // visionaray

#endif // VSNRAY_SURFACE_INTERACTION_H
