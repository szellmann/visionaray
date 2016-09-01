// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <array>
#include <cstddef>

#include <visionaray/material.h>

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

template <size_t N, typename ...Ts>
class generic_material;

//-------------------------------------------------------------------------------------------------
// SSE type used internally. Contains four generic materials
//

template <typename ...Ts>
class generic_material<4, Ts...>
{
public:

    using scalar_type      = simd::float4;
    using single_material  = visionaray::generic_material<Ts...>;

public:

    generic_material(std::array<single_material, 4> const& mats)
        : mats_(mats)
    {
    }

    single_material& get(int i)
    {
        return mats_[i];
    }

    single_material const& get(int i) const
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
        std::array<spectrum<float>, 4> amb;

        for (size_t i = 0; i < 4; ++i)
        {
            amb[i] = mats_[i].ambient();
        }

        return simd::pack(amb);
    }


    template <typename SR>
    spectrum<simd::float4> shade(SR const& sr) const
    {
        auto sr4 = unpack(sr);

        std::array<spectrum<float>, 4> shaded;

        for (size_t i = 0; i < 4; ++i)
        {
            shaded[i] = mats_[i].shade(sr4[i]);
        }

        return simd::pack(shaded);
    }

    template <typename SR, typename S /* sampler */>
    spectrum<simd::float4> sample(
            SR const&                   sr,
            vector<3, simd::float4>&    refl_dir,
            simd::float4&               pdf,
            S&                          samp
            ) const
    {
        using float_array = typename simd::aligned_array<simd::float4>::type;

        auto sr4 = unpack(sr);
        auto& s = samp.get_sampler();

        std::array<vector<3, float>, 4> rd4;
        float_array                     pdf4;
        std::array<spectrum<float>, 4>  sampled;

        for (size_t i = 0; i < 4; ++i)
        {
            sampled[i] = mats_[i].sample(sr4[i], rd4[i], pdf4[i], s);
        }

        refl_dir = simd::pack(rd4);
        pdf = simd::float4(pdf4);
        return simd::pack(sampled);
    }

private:

    std::array<single_material, 4> mats_;

};

#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

//-------------------------------------------------------------------------------------------------
// AVX type used internally. Contains eight generic materials
//

template <typename ...Ts>
class generic_material<8, Ts...>
{
public:

    using scalar_type     = simd::float8;
    using single_material = visionaray::generic_material<Ts...>;

public:

    generic_material(std::array<single_material, 8> const& mats)
        : mats_(mats)
    {
    }

    single_material& get(int i)
    {
        return mats_[i];
    }

    single_material const& get(int i) const
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
        std::array<spectrum<float>, 8> amb;

        for (size_t i = 0; i < 8; ++i)
        {
            amb[i] = mats_[i].ambient();
        }

        return simd::pack(amb);
    }


    template <typename SR>
    spectrum<simd::float8> shade(SR const& sr) const
    {
        auto sr8 = unpack(sr);

        std::array<spectrum<float>, 8> shaded;

        for (size_t i = 0; i < 8; ++i)
        {
            shaded[i] = mats_[i].shade(sr8[i]);
        }

        return simd::pack(shaded);
    }

    template <typename SR, typename S /* sampler */>
    spectrum<simd::float8> sample(
            SR const&                   sr,
            vector<3, simd::float8>&    refl_dir,
            simd::float8&               pdf,
            S&                          samp
            ) const
    {
        using float_array = typename simd::aligned_array<simd::float8>::type;

        auto sr8 = unpack(sr);
        auto& s = samp.get_sampler();

        std::array<vector<3, float>, 8> rd8;
        float_array                     pdf8;
        std::array<spectrum<float>, 8>  sampled;

        for (size_t i = 0; i < 8; ++i)
        {
            sampled[i] = mats_[i].sample(sr8[i], rd8[i], pdf8[i], s);
        }

        refl_dir = simd::pack(rd8);
        pdf = simd::float8(pdf8);
        return simd::pack(sampled);
    }

private:

    std::array<single_material, 8> mats_;

};

#endif // VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX


//-------------------------------------------------------------------------------------------------
// Pack and unpack
//

template <typename ...Ts>
inline generic_material<4, Ts...> pack(std::array<visionaray::generic_material<Ts...>, 4> const& mats)
{
    return generic_material<4, Ts...>(mats);
}

template <typename ...Ts>
inline std::array<visionaray::generic_material<Ts...>, 4> unpack(generic_material<4, Ts...> const& m4)
{
    return std::array<visionaray::generic_material<Ts...>, 4>{{ m4.get(0), m4.get(1), m4.get(2), m4.get(3) }};
}

#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

template <typename ...Ts>
inline generic_material<8, Ts...> pack(std::array<visionaray::generic_material<Ts...>, 8> const& mats)
{
    return generic_material<8, Ts...>(mats);
}

template <typename ...Ts>
inline std::array<visionaray::generic_material<Ts...>, 8> unpack(generic_material<8, Ts...> const& m8)
{
    return std::array<visionaray::generic_material<Ts...>, 8>{{
            m8.get(0), m8.get(1), m8.get(2), m8.get(3),
            m8.get(4), m8.get(5), m8.get(6), m8.get(7)
            }};
}

#endif // VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

} // simd
} // visionaray
