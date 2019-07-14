// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_SPD_MEASURED_H
#define VSNRAY_DETAIL_SPD_MEASURED_H 1

#include <algorithm>
#include <cassert>

#include <utility>
#include <vector>

#include <visionaray/math/math.h>

#include "../macros.h"

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
//
//

class measured
{
public:

    // Construct from separate wavelength/value arrays
    measured(float const* wavelengths, float const* values, size_t num)
        : values_(num)
    {
        for (size_t i = 0; i < num; ++i)
        {
            values_[i] = { wavelengths[i], values[i] };
        }
    }

    // Construct from wavelength/value pairs
    measured(std::vector<std::pair<float, float>> const& values)
        : values_(values)
    {
    }

    VSNRAY_FUNC float operator()(float lambda) const
    {
        // Require values to be sorted by wavelength
        assert(std::is_sorted(
                values.begin(),
                values.end(),
                [](std::pair<float, float> a, std::pair<float, float> b)
                {
                    return a.first < b.first;
                }
                ));

        // Require lambda to be contained in the right open interval
        // defined by the sorted wavelengths
        assert(values_.front().first >= lambda && values_.back().first < lambda);

        // Find the interval that contains lambda
        // TODO: binary search!
        size_t ival(-1);
        for (size_t i = 0; i < values_.size() - 1; ++i)
        {
            if (values_[i].first >= lambda && values_[i + 1].first)
            {
                ival = i;
                break;
            }
        }

        float x = (lambda - values_[ival].first) / (values_[ival + 1].first - values_[ival].first);
        return lerp(values_[ival].second, values_[ival + 1].second, x);
    }

private:

    // wavelength (nm) / value
    std::vector<std::pair<float, float>> values_;

};

} // visionaray

#endif // VSNRAY_DETAIL_SPD_MEASURED_H
