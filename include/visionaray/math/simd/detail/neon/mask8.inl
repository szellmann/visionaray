// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <stdexcept>

namespace MATH_NAMESPACE
{
namespace simd
{

//-------------------------------------------------------------------------------------------------
// mask8 members
//

VSNRAY_FORCE_INLINE mask8::basic_mask(uint32x4_t const& i1, uint32x4_t const& i2)
{
    i[0] = i1;
    i[1] = i2;
}

VSNRAY_FORCE_INLINE mask8::basic_mask(int8 const& m)
{
    i[0] = vreinterpretq_u32_s32(m.value[0]);
    i[1] = vreinterpretq_u32_s32(m.value[1]);
}

VSNRAY_FORCE_INLINE mask8::basic_mask(bool b)
{
    i[0] = vdupq_n_u32(b ? 0xFFFFFFFF : 0x00000000);
    i[1] = vdupq_n_u32(b ? 0xFFFFFFFF : 0x00000000);
}


//-------------------------------------------------------------------------------------------------
// Static cast
//

VSNRAY_FORCE_INLINE int8 convert_to_int(mask8 const& a)
{
    return int8((int32x4_t)a.i[0], (int32x4_t)a.i[1]);
}


//-------------------------------------------------------------------------------------------------
// any / all intrinsics
//

VSNRAY_FORCE_INLINE bool any(mask8 const& m)
{
    auto u = reinterpret_cast<unsigned const*>(&m);
    return u[0] || u[1] || u[2] || u[3] || u[4] || u[5] || u[6] || u[7];
}

VSNRAY_FORCE_INLINE bool all(mask8 const& m)
{
    auto u = reinterpret_cast<unsigned const*>(&m);
    return u[0] && u[1] && u[2] && u[3] && u[4] && u[5] && u[6] && u[7];
}


//-------------------------------------------------------------------------------------------------
// select intrinsic
//

VSNRAY_FORCE_INLINE mask8 select(mask8 const& m, mask8 const& a, mask8 const& b)
{
    return mask8(
        vbslq_u32(m.i[0], a.i[0], b.i[0]),
        vbslq_u32(m.i[1], a.i[1], b.i[1])
        );
}


//-------------------------------------------------------------------------------------------------
// Load / store
//


//-------------------------------------------------------------------------------------------------
// Bitwise operations
//

VSNRAY_FORCE_INLINE mask8 operator!(mask8 const& a)
{
    return mask8(vmvnq_u32(a.i[0]), vmvnq_u32(a.i[1]));
}

VSNRAY_FORCE_INLINE mask8 operator&(mask8 const& a, mask8 const& b)
{
    return mask8(vandq_u32(a.i[0], b.i[0]), vandq_u32(a.i[1], b.i[1]));
}

VSNRAY_FORCE_INLINE mask8 operator|(mask8 const& a, mask8 const& b)
{
    return mask8(vorrq_u32(a.i[0], b.i[0]), vorrq_u32(a.i[1], b.i[1]));
}

VSNRAY_FORCE_INLINE mask8 operator^(mask8 const& a, mask8 const& b)
{
    return mask8(veorq_u32(a.i[0], b.i[0]), veorq_u32(a.i[1], b.i[1]));
}


//-------------------------------------------------------------------------------------------------
// Logical operations
//

VSNRAY_FORCE_INLINE mask8 operator&&(mask8 const& a, mask8 const& b)
{
    // Ok because masks only store booleans
    return a & b;
}

VSNRAY_FORCE_INLINE mask8 operator||(mask8 const& a, mask8 const& b)
{
    // Ok because masks only store booleans
    return a | b;
}


//-------------------------------------------------------------------------------------------------
// Comparisons
//

VSNRAY_FORCE_INLINE mask8 operator==(mask8 const& u, mask8 const& v)
{
    return mask8(vceqq_u32(u.i[0], v.i[0]), vceqq_u32(u.i[1], v.i[1]));
}

VSNRAY_FORCE_INLINE mask8 operator!=(mask8 const& u, mask8 const& v)
{
    return mask8(
        vmvnq_u32(vceqq_u32(u.i[0], v.i[0])),
        vmvnq_u32(vceqq_u32(u.i[1], v.i[1]))
        );
}

} // simd


//-------------------------------------------------------------------------------------------------
// Import SIMD intrinsics into namespace visionaray.
// Enable ADL!
//

using simd::select;
using simd::store;
using simd::any;
using simd::all;

} // MATH_NAMESPACE
