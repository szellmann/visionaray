// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <array>

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

    generic_material4(std::array<generic_material<Ts...>, 4> const& mats)
        : mats_(mats)
    {
    }

    generic_material<Ts...>& get(int i)
    {
        return mats_[i];
    }

    generic_material<Ts...> const& get(int i) const
    {
        return mats_[i];
    }

    simd::mask4 is_emissive() const
    {
        return simd::mask4(
                mats_[0].is_emissive(),
                mats_[1].is_emissive(),
                mats_[2].is_emissive(),
                mats_[3].is_emissive()
                );
    }

    spectrum<simd::float4> ambient() const
    {
        return simd::pack(
                spectrum<float>( mats_[0].ambient() ),
                spectrum<float>( mats_[1].ambient() ),
                spectrum<float>( mats_[2].ambient() ),
                spectrum<float>( mats_[3].ambient() )
                );
    }


    template <typename SR>
    spectrum<simd::float4> shade(SR const& sr) const
    {
        auto sr4 = unpack(sr);
        return simd::pack(
                mats_[0].shade(sr4[0]),
                mats_[1].shade(sr4[1]),
                mats_[2].shade(sr4[2]),
                mats_[3].shade(sr4[3])
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
            spectrum<float>( mats_[0].sample(sr4[0], rd4[0], pdf4[0], s) ),
            spectrum<float>( mats_[1].sample(sr4[1], rd4[1], pdf4[1], s) ),
            spectrum<float>( mats_[2].sample(sr4[2], rd4[2], pdf4[2], s) ),
            spectrum<float>( mats_[3].sample(sr4[3], rd4[3], pdf4[3], s) )
        };
        refl_dir = simd::pack( rd4[0], rd4[1], rd4[2], rd4[3] );
        pdf = simd::float4(pdf4);
        return simd::pack( v[0], v[1], v[2], v[3] );
    }

private:

    std::array<generic_material<Ts...>, 4> mats_;

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

    generic_material8(std::array<generic_material<Ts...>, 8> const& mats)
        : mats_(mats)
    {
    }

    generic_material<Ts...>& get(int i)
    {
        return mats_[i];
    }

    generic_material<Ts...> const& get(int i) const
    {
        return mats_[i];
    }

    simd::mask8 is_emissive() const
    {
        return simd::mask8(
                mats_[0].is_emissive(),
                mats_[1].is_emissive(),
                mats_[2].is_emissive(),
                mats_[3].is_emissive(),
                mats_[4].is_emissive(),
                mats_[5].is_emissive(),
                mats_[6].is_emissive(),
                mats_[7].is_emissive()
                );
    }

    spectrum<simd::float8> ambient() const
    {
        return simd::pack(
                spectrum<float>( mats_[0].ambient() ),
                spectrum<float>( mats_[1].ambient() ),
                spectrum<float>( mats_[2].ambient() ),
                spectrum<float>( mats_[3].ambient() ),
                spectrum<float>( mats_[4].ambient() ),
                spectrum<float>( mats_[5].ambient() ),
                spectrum<float>( mats_[6].ambient() ),
                spectrum<float>( mats_[7].ambient() )
                );
    }


    template <typename SR>
    spectrum<simd::float8> shade(SR const& sr) const
    {
        auto sr8 = unpack(sr);
        return simd::pack(
                mats_[0].shade(sr8[0]),
                mats_[1].shade(sr8[1]),
                mats_[2].shade(sr8[2]),
                mats_[3].shade(sr8[3]),
                mats_[4].shade(sr8[4]),
                mats_[5].shade(sr8[5]),
                mats_[6].shade(sr8[6]),
                mats_[7].shade(sr8[7])
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
            spectrum<float>( mats_[0].sample(sr8[0], rd8[0], pdf8[0], s) ),
            spectrum<float>( mats_[1].sample(sr8[1], rd8[1], pdf8[1], s) ),
            spectrum<float>( mats_[2].sample(sr8[2], rd8[2], pdf8[2], s) ),
            spectrum<float>( mats_[3].sample(sr8[3], rd8[3], pdf8[3], s) ),
            spectrum<float>( mats_[4].sample(sr8[4], rd8[4], pdf8[4], s) ),
            spectrum<float>( mats_[5].sample(sr8[5], rd8[5], pdf8[5], s) ),
            spectrum<float>( mats_[6].sample(sr8[6], rd8[6], pdf8[6], s) ),
            spectrum<float>( mats_[7].sample(sr8[7], rd8[7], pdf8[7], s) )
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

    std::array<generic_material<Ts...>, 8> mats_;

};

#endif // VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX


//-------------------------------------------------------------------------------------------------
// Pack and unpack
//

template <typename ...Ts>
inline generic_material4<Ts...> pack(std::array<generic_material<Ts...>, 4> const& mats)
{
    return generic_material4<Ts...>(mats);
}

template <typename ...Ts>
inline std::array<generic_material<Ts...>, 4> unpack(generic_material4<Ts...> const& m4)
{
    return std::array<generic_material<Ts...>, 4>{{ m4.get(0), m4.get(1), m4.get(2), m4.get(3) }};
}

#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

template <typename ...Ts>
inline generic_material8<Ts...> pack(std::array<generic_material<Ts...>, 8> const& mats)
{
    return generic_material8<Ts...>(mats);
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
