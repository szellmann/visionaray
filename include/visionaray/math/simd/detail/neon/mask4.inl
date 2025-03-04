// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <stdexcept>

namespace MATH_NAMESPACE
{
namespace simd
{

//-------------------------------------------------------------------------------------------------
// mask4 members
//

VSNRAY_FORCE_INLINE mask4::basic_mask(uint32x4_t const& m)
    : i(m)
{
}

VSNRAY_FORCE_INLINE mask4::basic_mask(int32x4_t const& m)
    : i(vreinterpretq_u32_s32(m))
{
}

VSNRAY_FORCE_INLINE mask4::basic_mask(bool x, bool y, bool z, bool w)
{
    VSNRAY_ALIGN(16) unsigned data[4] = {
            x ? 0xFFFFFFFF : 0x00000000,
            y ? 0xFFFFFFFF : 0x00000000,
            z ? 0xFFFFFFFF : 0x00000000,
            w ? 0xFFFFFFFF : 0x00000000
            };
    i = vld1q_u32(data);
}

VSNRAY_FORCE_INLINE mask4::basic_mask(bool const v[4])
{
    VSNRAY_ALIGN(16) unsigned data[4] = {
            v[0] ? 0xFFFFFFFF : 0x00000000,
            v[1] ? 0xFFFFFFFF : 0x00000000,
            v[2] ? 0xFFFFFFFF : 0x00000000,
            v[3] ? 0xFFFFFFFF : 0x00000000
            };
    i = vld1q_u32(data);
}

VSNRAY_FORCE_INLINE mask4::basic_mask(bool b)
    : i(vdupq_n_u32(b ? 0xFFFFFFFF : 0x00000000))
{
}


//-------------------------------------------------------------------------------------------------
// Static cast
//

VSNRAY_FORCE_INLINE int4 convert_to_int(mask4 const& a)
{
    return (int32x4_t)a.i;
}


//-------------------------------------------------------------------------------------------------
// any / all intrinsics
//

VSNRAY_FORCE_INLINE bool any(mask4 const& m)
{
    auto u = reinterpret_cast<unsigned const*>(&m);
    return u[0] || u[1] || u[2] || u[3];
}

VSNRAY_FORCE_INLINE bool all(mask4 const& m)
{
    auto u = reinterpret_cast<unsigned const*>(&m);
    return u[0] && u[1] && u[2] && u[3];
}


//-------------------------------------------------------------------------------------------------
// select intrinsic
//

VSNRAY_FORCE_INLINE mask4 select(mask4 const& m, mask4 const& a, mask4 const& b)
{
    return vbslq_u32(m.i, a.i, b.i);
}


//-------------------------------------------------------------------------------------------------
// Load / store
//


//-------------------------------------------------------------------------------------------------
// Bitwise operations
//

VSNRAY_FORCE_INLINE mask4 operator!(mask4 const& a)
{
    return vmvnq_u32(a.i);
}

VSNRAY_FORCE_INLINE mask4 operator&(mask4 const& a, mask4 const& b)
{
    return vandq_u32(a.i, b.i);
}

VSNRAY_FORCE_INLINE mask4 operator|(mask4 const& a, mask4 const& b)
{
    return vorrq_u32(a.i, b.i);
}

VSNRAY_FORCE_INLINE mask4 operator^(mask4 const& a, mask4 const& b)
{
    return veorq_u32(a.i, b.i);
}


//-------------------------------------------------------------------------------------------------
// Logical operations
//

VSNRAY_FORCE_INLINE mask4 operator&&(mask4 const& a, mask4 const& b)
{
    // Ok because masks only store booleans
    return a & b;
}

VSNRAY_FORCE_INLINE mask4 operator||(mask4 const& a, mask4 const& b)
{
    // Ok because masks only store booleans
    return a | b;
}


//-------------------------------------------------------------------------------------------------
// Comparisons
//

VSNRAY_FORCE_INLINE mask4 operator==(mask4 const& u, mask4 const& v)
{
    return vceqq_u32(u.i, v.i);
}

VSNRAY_FORCE_INLINE mask4 operator!=(mask4 const& u, mask4 const& v)
{
    return vmvnq_u32(vceqq_u32(u.i, v.i));
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
