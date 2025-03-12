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
        static const float table[54] = {
            0.0003, // 300
            0.0336, // 310
            0.2042, // 320
            0.3742, // 330
            0.4029, // 340
            0.4524, // 350
            0.4692, // 360
            0.5241, // 370
            0.5088, // 380
            0.5483, // 390
            0.8293, // 400
            0.9169, // 410
            0.9362, // 420
            0.8683, // 430
            1.0498, // 440
            1.1711, // 450
            1.1791, // 460
            1.1494, // 470
            1.1598, // 480
            1.0887, // 490
            1.0940, // 500
            1.0784, // 510
            1.0481, // 520
            1.0770, // 530
            1.0441, // 540
            1.0405, // 550
            1.0000, // 560
            0.9633, // 570
            0.9578, // 580
            0.8871, // 590
            0.9004, // 600
            0.8965, // 610
            0.8775, // 620
            0.8335, // 630
            0.8378, // 640
            0.8013, // 650
            0.8033, // 660
            0.8241, // 670
            0.7843, // 680
            0.6983, // 690
            0.7174, // 700
            0.7446, // 710
            0.6170, // 720
            0.6999, // 730
            0.7519, // 740
            0.6368, // 750
            0.4649, // 760
            0.6690, // 770
            0.6347, // 780
            0.6440, // 790
            0.5954, // 800
            0.5203, // 810
            0.5752, // 820
            0.6040  // 830
        };

        if (lambda < 300 || lambda >= 830)
        {
            return 0;
        }

        int i = static_cast<int>(floor(lambda)) / 10;
        float s = (lambda - i * 10.0f) / 10.0f;

        i -= 30;

        return lerp_r(table[i], table[i + 1], s);
    }
};

} // visionaray

#endif // VSNRAY_DETAIL_SPD_D65_H
