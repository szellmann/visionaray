// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <array>
#include <cassert>

#include <visionaray/material.h>

#pragma once

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Public interface
//

template <typename T, typename ...Ts>
template <template <typename> class M>
inline generic_material<T, Ts...>::generic_material(M<typename T::scalar_type> const& material)
    : generic_material<T, Ts...>::base_type(material)
{
}

template <typename T, typename ...Ts>
VSNRAY_FUNC
inline bool generic_material<T, Ts...>::is_emissive() const
{
    return apply_visitor( is_emissive_visitor(), *this );
}

template <typename T, typename ...Ts>
VSNRAY_FUNC
inline spectrum<typename T::scalar_type> generic_material<T, Ts...>::ambient() const
{
    return apply_visitor( ambient_visitor(), *this );
}

template <typename T, typename ...Ts>
template <typename SR>
VSNRAY_FUNC
inline spectrum<typename SR::scalar_type> generic_material<T, Ts...>::shade(SR const& sr) const
{
    return apply_visitor( shade_visitor<SR>(sr), *this );
}

template <typename T, typename ...Ts>
template <typename SR, typename U, typename Sampler>
VSNRAY_FUNC
inline spectrum<U> generic_material<T, Ts...>::sample(
        SR const&       sr,
        vector<3, U>&   refl_dir,
        U&              pdf,
        Sampler&        sampler
        ) const
{
    return apply_visitor( sample_visitor<SR, U, Sampler>(sr, refl_dir, pdf, sampler), *this );
}


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
template <typename SR, typename U, typename Sampler>
struct generic_material<T, Ts...>::sample_visitor
{
    using return_type = spectrum<typename SR::scalar_type>;

    VSNRAY_FUNC
    sample_visitor(SR const& sr, vector<3, U>& refl_dir, U& pdf, Sampler& sampler)
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
    Sampler&        sampler_;
};


namespace simd
{

//-------------------------------------------------------------------------------------------------
// SSE type used internally. Contains four generic materials
//

template <typename ...Ts>
class generic_material4
{
public:

    using scalar_type = simd::float4;

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

    generic_material<Ts...>& get(int i)
    {
        assert( i >= 0 && i < 4 );

        return reinterpret_cast<generic_material<Ts...>*>(this)[i];
    }

    generic_material<Ts...> const& get(int i) const
    {
        assert( i >= 0 && i < 4 );

        return reinterpret_cast<generic_material<Ts...> const*>(this)[i];
    }

    simd::mask4 is_emissive() const
    {
        return simd::mask4(
                m1_.is_emissive(),
                m2_.is_emissive(),
                m3_.is_emissive(),
                m4_.is_emissive()
                );
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
        auto sr4 = unpack(sr);
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
        auto sr4 = unpack(sr);
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

#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

//-------------------------------------------------------------------------------------------------
// AVX type used internally. Contains eight generic materials
//

template <typename ...Ts>
class generic_material8
{
public:

    using scalar_type = simd::float8;

public:

    generic_material8(
            generic_material<Ts...> const& m1,
            generic_material<Ts...> const& m2,
            generic_material<Ts...> const& m3,
            generic_material<Ts...> const& m4,
            generic_material<Ts...> const& m5,
            generic_material<Ts...> const& m6,
            generic_material<Ts...> const& m7,
            generic_material<Ts...> const& m8
            )
        : m1_(m1)
        , m2_(m2)
        , m3_(m3)
        , m4_(m4)
        , m5_(m5)
        , m6_(m6)
        , m7_(m7)
        , m8_(m8)
    {
    }

    generic_material<Ts...>& get(int i)
    {
        assert( i >= 0 && i < 8 );

        return reinterpret_cast<generic_material<Ts...>*>(this)[i];
    }

    generic_material<Ts...> const& get(int i) const
    {
        assert( i >= 0 && i < 8 );

        return reinterpret_cast<generic_material<Ts...> const*>(this)[i];
    }

    simd::mask8 is_emissive() const
    {
        return simd::mask8(
                m1_.is_emissive(),
                m2_.is_emissive(),
                m3_.is_emissive(),
                m4_.is_emissive(),
                m5_.is_emissive(),
                m6_.is_emissive(),
                m7_.is_emissive(),
                m8_.is_emissive()
                );
    }

    spectrum<simd::float8> ambient() const
    {
        return simd::pack(
                spectrum<float>( m1_.ambient() ),
                spectrum<float>( m2_.ambient() ),
                spectrum<float>( m3_.ambient() ),
                spectrum<float>( m4_.ambient() ),
                spectrum<float>( m5_.ambient() ),
                spectrum<float>( m6_.ambient() ),
                spectrum<float>( m7_.ambient() ),
                spectrum<float>( m8_.ambient() )
                );
    }


    template <typename SR>
    spectrum<simd::float8> shade(SR const& sr) const
    {
        auto sr8 = unpack(sr);
        return simd::pack(
                m1_.shade(sr8[0]),
                m2_.shade(sr8[1]),
                m3_.shade(sr8[2]),
                m4_.shade(sr8[3]),
                m5_.shade(sr8[4]),
                m6_.shade(sr8[5]),
                m7_.shade(sr8[6]),
                m8_.shade(sr8[7])
                );
    }

    template <typename SR, typename S /* sampler */>
    spectrum<simd::float8> sample(
            SR const&                   sr,
            vector<3, simd::float8>&    refl_dir,
            simd::float8&               pdf,
            S&                          samp
            ) const
    {
        auto sr8 = unpack(sr);
        vector<3, float> rd8[8];
        VSNRAY_ALIGN(32) float pdf8[] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
        auto& s = samp.get_sampler();
        spectrum<float> v[] =
        {
            spectrum<float>( m1_.sample(sr8[0], rd8[0], pdf8[0], s) ),
            spectrum<float>( m2_.sample(sr8[1], rd8[1], pdf8[1], s) ),
            spectrum<float>( m3_.sample(sr8[2], rd8[2], pdf8[2], s) ),
            spectrum<float>( m4_.sample(sr8[3], rd8[3], pdf8[3], s) ),
            spectrum<float>( m5_.sample(sr8[4], rd8[4], pdf8[4], s) ),
            spectrum<float>( m6_.sample(sr8[5], rd8[5], pdf8[5], s) ),
            spectrum<float>( m7_.sample(sr8[6], rd8[6], pdf8[6], s) ),
            spectrum<float>( m8_.sample(sr8[7], rd8[7], pdf8[7], s) )
        };
        refl_dir = simd::pack(
                rd8[0], rd8[1], rd8[2], rd8[3],
                rd8[4], rd8[5], rd8[6], rd8[7]
                );
        pdf = simd::float8(pdf8);
        return simd::pack(
                v[0], v[1], v[2], v[3],
                v[4], v[5], v[6], v[7]
                );
    }

private:

    generic_material<Ts...> m1_;
    generic_material<Ts...> m2_;
    generic_material<Ts...> m3_;
    generic_material<Ts...> m4_;
    generic_material<Ts...> m5_;
    generic_material<Ts...> m6_;
    generic_material<Ts...> m7_;
    generic_material<Ts...> m8_;

};

#endif // VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX


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

template <typename ...Ts>
inline std::array<generic_material<Ts...>, 4> unpack(generic_material4<Ts...> const& m4)
{
    return std::array<generic_material<Ts...>, 4>{{ m4.get(0), m4.get(1), m4.get(2), m4.get(3) }};
}

#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

template <typename ...Ts>
inline generic_material8<Ts...> pack(
        generic_material<Ts...> const& m1,
        generic_material<Ts...> const& m2,
        generic_material<Ts...> const& m3,
        generic_material<Ts...> const& m4,
        generic_material<Ts...> const& m5,
        generic_material<Ts...> const& m6,
        generic_material<Ts...> const& m7,
        generic_material<Ts...> const& m8
        )
{
    return generic_material8<Ts...>(m1, m2, m3, m4, m5, m6, m7, m8);
}

template <typename ...Ts>
inline std::array<generic_material<Ts...>, 8> unpack(generic_material8<Ts...> const& m8)
{
    return std::array<generic_material<Ts...>, 8>{{
            m8.get(0), m8.get(1), m8.get(2), m8.get(3),
            m8.get(4), m8.get(5), m8.get(6), m8.get(7)
            }};
}

#endif // VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

} // simd
} // visionaray
