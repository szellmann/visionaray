// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_PTEX_H
#define VSNRAY_COMMON_PTEX_H 1

#include <common/config.h>

#include <memory>
#include <string>

#if VSNRAY_COMMON_HAVE_PTEX
#include <Ptexture.h>
#endif

#include <visionaray/math/simd/type_traits.h>
#include <visionaray/math/triangle.h>

namespace visionaray
{
namespace ptex
{

struct texture
{
    std::string filename;
    std::shared_ptr<PtexPtr<PtexCache>> cache;
};


//-------------------------------------------------------------------------------------------------
// Wrapper for face id for ADL
//

class face_id_t
{
public:

    face_id_t() = default;

    /* implicit */ face_id_t(int v) : value_(v) {}

    operator int() const { return value_; }

private:

    int value_;
};


//-------------------------------------------------------------------------------------------------
// Tuple to access WDAS Ptex texture
//

template <typename T>
struct coordinate
{
    using I = simd::int_type_t<T>;

    I face_id;
    T u;
    T v;
    T du;
    T dv;
};

} // ptex
} // visionaray

#include "ptex.inl"

#endif // VSNRAY_COMMON_PTEX_H
