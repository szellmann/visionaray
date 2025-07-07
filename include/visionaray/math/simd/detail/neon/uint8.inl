// This file is distributed under the MIT license.
// See the LICENSE file for details.

namespace MATH_NAMESPACE
{
namespace simd
{

//-------------------------------------------------------------------------------------------------
// uint8 members
//

VSNRAY_FORCE_INLINE uint8::basic_uint(uint32x4_t const& v1, uint32x4_t const& v2)
{
    value[0] = v1;
    value[1] = v2;
}


//-------------------------------------------------------------------------------------------------
// Logical operations
//

VSNRAY_FORCE_INLINE uint8 operator&&(uint8 const& u, uint8 const& v)
{
    return uint8(vandq_u32(u.value[0], v.value[0]), vandq_u32(u.value[1], v.value[1]));
}

VSNRAY_FORCE_INLINE uint8 operator||(uint8 const& u, uint8 const& v)
{
    return uint8(vorrq_u32(u.value[0], v.value[0]), vorrq_u32(u.value[1], v.value[1]));
}


//-------------------------------------------------------------------------------------------------
// Comparisons
//

VSNRAY_FORCE_INLINE mask8 operator<(uint8 const& u, uint8 const& v)
{
    return mask8(vcltq_u32(u.value[0], v.value[0]), vcltq_u32(u.value[1], v.value[1]));
}

VSNRAY_FORCE_INLINE mask8 operator>(uint8 const& u, uint8 const& v)
{
    return mask8(vcgtq_u32(u.value[0], v.value[0]), vcgtq_u32(u.value[1], v.value[1]));
}

VSNRAY_FORCE_INLINE mask8 operator<=(uint8 const& u, uint8 const& v)
{
    return mask8(vcleq_u32(u.value[0], v.value[0]), vcleq_u32(u.value[1], v.value[1]));
}

VSNRAY_FORCE_INLINE mask8 operator>=(uint8 const& u, uint8 const& v)
{
    return mask8(vcgeq_u32(u.value[0], v.value[0]), vcgeq_u32(u.value[1], v.value[1]));
}

VSNRAY_FORCE_INLINE mask8 operator==(uint8 const& u, uint8 const& v)
{
    return mask8(vceqq_u32(u.value[0], v.value[0]), vceqq_u32(u.value[1], v.value[1]));
}

VSNRAY_FORCE_INLINE mask8 operator!=(uint8 const& u, uint8 const& v)
{
    return mask8(
        vmvnq_u32(vceqq_u32(u.value[0], v.value[0])),
        vmvnq_u32(vceqq_u32(u.value[1], v.value[1]))
        );
}

} // simd
} // MATH_NAMESPACE
