// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <visionaray/material.h>

#pragma once

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Private variant visitors
//

template <typename T, typename ...Ts>
struct generic_material<T, Ts...>::is_emissive_visitor
{
    using Base = generic_material<T, Ts...>;
    using return_type = bool;

    template <typename X>
    VSNRAY_FUNC
    bool operator()(X) const
    {
        return false;
    }

    VSNRAY_FUNC
    bool operator()(emissive<typename Base::scalar_type>) const
    {
        return true;
    }
};

template <typename T, typename ...Ts>
struct generic_material<T, Ts...>::ambient_visitor
{
    using Base = generic_material<T, Ts...>;
    using return_type = spectrum<typename Base::scalar_type>;

    template <typename X>
    VSNRAY_FUNC
    return_type operator()(X const& ref) const
    {
        return ref.ambient();
    }
};

template <typename T, typename ...Ts>
template <typename SR>
struct generic_material<T, Ts...>::shade_visitor
{
    using return_type = spectrum<typename SR::scalar_type>;

    VSNRAY_FUNC
    shade_visitor(SR const& sr) : sr_(sr) {}

    template <typename X>
    VSNRAY_FUNC
    return_type operator()(X const& ref) const
    {
        return ref.shade(sr_);
    }

    SR const& sr_;
};

template <typename T, typename ...Ts>
template <typename SR, typename U, typename S>
struct generic_material<T, Ts...>::sample_visitor
{
    using return_type = spectrum<typename SR::scalar_type>;

    VSNRAY_FUNC
    sample_visitor(SR const& sr, vector<3, U>& refl_dir, U& pdf, S& sampler)
        : sr_(sr)
        , refl_dir_(refl_dir)
        , pdf_(pdf)
        , sampler_(sampler)
    {
    }

    template <typename X>
    VSNRAY_FUNC
    return_type operator()(X const& ref) const
    {
        return ref.sample(sr_, refl_dir_, pdf_, sampler_);
    }

    SR const&       sr_;
    vector<3, U>&   refl_dir_;
    U&              pdf_;
    S&              sampler_;
};


//-------------------------------------------------------------------------------------------------
// SSE type used internally. Contains four generic materials
//

namespace simd
{

template <typename ...Ts>
class generic_material4
{
public:

    using scalar_type   = simd::float4;

public:

    generic_material4(
            generic_material<Ts...> const& m1,
            generic_material<Ts...> const& m2,
            generic_material<Ts...> const& m3,
            generic_material<Ts...> const& m4
            )
        : m1_(m1)
        , m2_(m2)
        , m3_(m3)
        , m4_(m4)
    {
    }

    simd::mask4 is_emissive() const
    {
        return simd::mask4( simd::int4(
                m1_.is_emissive() ? 0xFFFFFFFF : 0x00000000,
                m2_.is_emissive() ? 0xFFFFFFFF : 0x00000000,
                m3_.is_emissive() ? 0xFFFFFFFF : 0x00000000,
                m4_.is_emissive() ? 0xFFFFFFFF : 0x00000000
                ) );
    }

    spectrum<simd::float4> ambient() const
    {
        return simd::pack(
                spectrum<float>( m1_.ambient() ),
                spectrum<float>( m2_.ambient() ),
                spectrum<float>( m3_.ambient() ),
                spectrum<float>( m4_.ambient() )
                );
    }


    template <typename SR>
    spectrum<simd::float4> shade(SR const& sr) const
    {
        auto sr4 = simd::unpack(sr);
        return simd::pack(
                m1_.shade(sr4[0]),
                m2_.shade(sr4[1]),
                m3_.shade(sr4[2]),
                m4_.shade(sr4[3])
                );
    }

    template <typename SR, typename S /* sampler */>
    spectrum<simd::float4> sample(
            SR const&                   sr,
            vector<3, simd::float4>&    refl_dir,
            simd::float4&               pdf,
            S&                          samp
            ) const
    {
        auto sr4 = simd::unpack(sr);
        vector<3, float> rd4[4];
        VSNRAY_ALIGN(16) float pdf4[] = { 0.0f, 0.0f, 0.0f, 0.0f };
        auto& s = samp.get_sampler();
        spectrum<float> v[] =
        {
            spectrum<float>( m1_.sample(sr4[0], rd4[0], pdf4[0], s) ),
            spectrum<float>( m2_.sample(sr4[1], rd4[1], pdf4[1], s) ),
            spectrum<float>( m3_.sample(sr4[2], rd4[2], pdf4[2], s) ),
            spectrum<float>( m4_.sample(sr4[3], rd4[3], pdf4[3], s) )
        };
        refl_dir = simd::pack( rd4[0], rd4[1], rd4[2], rd4[3] );
        pdf = simd::float4(pdf4);
        return simd::pack( v[0], v[1], v[2], v[3] );
    }

private:

    generic_material<Ts...> m1_;
    generic_material<Ts...> m2_;
    generic_material<Ts...> m3_;
    generic_material<Ts...> m4_;

};


//-------------------------------------------------------------------------------------------------
// Pack and unpack
//

template <typename ...Ts>
inline generic_material4<Ts...> pack(
        generic_material<Ts...> const& m1,
        generic_material<Ts...> const& m2,
        generic_material<Ts...> const& m3,
        generic_material<Ts...> const& m4
        )
{
    return generic_material4<Ts...>(m1, m2, m3, m4);
}

// TODO: unpack

} // simd
} // visionaray
