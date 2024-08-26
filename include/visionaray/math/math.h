// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_MATH_H
#define VSNRAY_MATH_H 1

#ifndef VSNRAY_NO_SIMD
#include "simd/simd.h"
#endif
#include "detail/math.h"

#include "aabb.h"
#include "axis.h"
#include "constants.h"
#include "coordinates.h"
#include "cylinder.h"
#include "fixed.h"
#include "intersect.h"
#include "interval.h"
#include "io.h"
#include "limits.h"
#include "matrix.h"
#include "norm.h"
#include "plane.h"
#include "primitive.h"
#include "project.h"
#include "quaternion.h"
#include "ray.h"
#include "rectangle.h"
#include "snorm.h"
#include "sphere.h"
#include "triangle.h"
#include "unorm.h"
#include "vector.h"

#endif // VSNRAY_MATH_H
