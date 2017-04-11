// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_SPD_D65_H
#define VSNRAY_DETAIL_SPD_D65_H 1

#include "../../math/vector.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Spectral power distribution of D65 illuminant (daylight 6500 K)
// Data derived from here: http://www.rit-mcsl.org/UsefulData/DaylightSeries.xls
// Normalized such that P(560) = 1
//

class spd_d65
{
public:

    VSNRAY_FUNC float operator()(float lambda) const
    {
        if (lambda < 300 || lambda >= 830)
        {
            return 0;
        }

        auto i = (int)floor(lambda) / 10;
        auto s = (lambda - i * 10.0f) / 10.0f;

        i -= 30;

        return lerp( table_[i], table_[i + 1], s );
    }

private:

    static const float table_[54];

};

} // visionaray

#endif // VSNRAY_DETAIL_SPD_D65_H
