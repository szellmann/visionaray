// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_TRAVERSE_H
#define VSNRAY_TRAVERSE_H

#include "math/math.h"

namespace visionaray
{


//-------------------------------------------------------------------------------------------------
// Intersect linear container
//

template <typename R, typename P>
VSNRAY_FUNC
hit_record<R, primitive<unsigned>> any_hit(R const& r, P begin, P end);

template <typename R, typename P>
VSNRAY_FUNC
hit_record<R, primitive<unsigned>> closest_hit(R const& r, P begin, P end);


} // visionaray

#include "detail/traverse_linear.inl"

#endif // VSNRAY_TRAVERSE_H
