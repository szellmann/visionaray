// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_WHITTED_INL
#define VSNRAY_WHITTED_INL 1

#include <array>

#include <visionaray/get_surface.h>
#include <visionaray/result_record.h>
#include <visionaray/traverse.h>

namespace visionaray
{
namespace whitted
{

namespace detail
{

//-------------------------------------------------------------------------------------------------
// TODO: consolidate this with "real" brdf sampling
// TODO: user should be able to customize this behavior
//

template <typename Vec3, typename Scalar>
struct bounce_result
{
    Vec3 reflected_dir;
    Vec3 refracted_dir;

    Scalar kr;
    Scalar kt;
};


// reflection
template <typename V, typename S>
VSNRAY_FUNC
inline auto make_bounce_result(V const& reflected_dir, S kr)
    -> bounce_result<V, S>
{
    return {
        reflected_dir,
        V(),
        kr,
        0.0f
        };
}

// reflection and refraction
template <typename V, typename S>
VSNRAY_FUNC
inline auto make_bounce_result(
        V const& reflected_dir,
        V const& refracted_dir,
        S kr,
        S kt
        )
    -> bounce_result<V, S>
{
    return {
        reflected_dir,
        refracted_dir,
        kr,
        kt
        };
}


//-------------------------------------------------------------------------------------------------
// specular_bounce() overloads for some materials
//

// fall-through, e.g. for plastic, assigns a dflt. reflectivity and no refraction
template <typename V, typename M>
VSNRAY_FUNC
inline auto specular_bounce(
        M const&    mat,
        V const&    view_dir,
        V const&    normal
        )
    -> decltype( make_bounce_result(V(), typename V::value_type()) )
{
    VSNRAY_UNUSED(mat);

    return make_bounce_result(
        reflect(view_dir, normal),
        typename V::value_type(0.1)
        );
}

// matte, no specular reflectivity, returns an arbitrary direction
template <typename V, typename S>
VSNRAY_FUNC
inline auto specular_bounce(
        matte<S> const& mat,
        V const&        view_dir,
        V const&        normal
        )
    -> decltype( make_bounce_result(V(), typename V::value_type()) )
{
    VSNRAY_UNUSED(mat);
    VSNRAY_UNUSED(view_dir);
    VSNRAY_UNUSED(normal);

    return make_bounce_result(
        V(),
        typename V::value_type(0.0)
        );
}

// mirror material, here we know kr
template <typename V, typename S>
VSNRAY_FUNC
inline auto specular_bounce(
        mirror<S> const&    mat,
        V const&            view_dir,
        V const&            normal
        )
    -> decltype( make_bounce_result(V(), typename V::value_type()) )
{
    return make_bounce_result(
        reflect(view_dir, normal),
        mat.get_kr()
        );
}


//-------------------------------------------------------------------------------------------------
// some special treatment for generic materials
//

template <typename V>
struct visitor
{
    using return_type = decltype( make_bounce_result(V(), typename V::value_type()) );

    VSNRAY_FUNC visitor(V const& vd, V const& n)
        : view_dir(vd)
        , normal(n)
    {
    }

    template <typename X>
    VSNRAY_FUNC
    return_type operator()(X const& ref) const
    {
        return specular_bounce(ref, view_dir, normal);
    }

    V const&    view_dir;
    V const&    normal;
};

template <typename V, typename ...Ts>
VSNRAY_FUNC
inline auto specular_bounce(
        generic_material<Ts...> const&  mat,
        V const&                        view_dir,
        V const&                        normal
        )
    -> decltype( make_bounce_result(V(), typename V::value_type()) )
{
    return apply_visitor(visitor<V>(view_dir, normal), mat);
}

template <typename ...Ts>
inline auto specular_bounce(
        simd::generic_material4<Ts...> const&   mat,
        vector<3, simd::float4> const&          view_dir,
        vector<3, simd::float4> const&          normal
        )
    -> bounce_result<vector<3, simd::float4>, simd::float4>
{
    using float_array = typename simd::aligned_array<simd::float4>::type;

    auto m4  = unpack(mat);
    auto vd4 = unpack(view_dir);
    auto n4  = unpack(normal);

    std::array<vector<3, float>, 4> refl_dir;
    std::array<vector<3, float>, 4> refr_dir;
    float_array                     kr;
    float_array                     kt;

    for (size_t i = 0; i < 4; ++i)
    {
        auto res = specular_bounce(m4[i], vd4[i], n4[i]);
        refl_dir[i] = res.reflected_dir;
        refr_dir[i] = res.refracted_dir;
        kr[i]       = res.kr;
        kt[i]       = res.kt;
    }

    return make_bounce_result(
        simd::pack(refl_dir),
        simd::pack(refr_dir),
        simd::float4(kr),
        simd::float4(kt)
        );
}

#if VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

template <typename ...Ts>
inline auto specular_bounce(
        simd::generic_material8<Ts...> const&   mat,
        vector<3, simd::float8> const&          view_dir,
        vector<3, simd::float8> const&          normal
        )
    -> bounce_result<vector<3, simd::float8>, simd::float8>
{
    using float_array = typename simd::aligned_array<simd::float8>::type;

    auto m8  = unpack(mat);
    auto vd8 = unpack(view_dir);
    auto n8  = unpack(normal);

    std::array<vector<3, float>, 8> refl_dir;
    std::array<vector<3, float>, 8> refr_dir;
    float_array                     kr;
    float_array                     kt;

    for (size_t i = 0; i < 8; ++i)
    {
        auto res = specular_bounce(m8[i], vd8[i], n8[i]);
        refl_dir[i] = res.reflected_dir;
        refr_dir[i] = res.refracted_dir;
        kr[i]       = res.kr;
        kt[i]       = res.kt;
    }

    return make_bounce_result(
        simd::pack(refl_dir),
        simd::pack(refr_dir),
        simd::float8(kr),
        simd::float8(kt)
        );
}

#endif // VSNRAY_SIMD_ISA >= VSNRAY_SIMD_ISA_AVX

} // detail


//-------------------------------------------------------------------------------------------------
// Whitted kernel
//

template <typename Params>
struct kernel
{

    Params params;

    template <typename Intersector, typename R>
    VSNRAY_FUNC result_record<typename R::scalar_type> operator()(Intersector& isect, R ray) const
    {

        using S = typename R::scalar_type;
        using V = typename result_record<S>::vec_type;
        using C = spectrum<S>;

        result_record<S> result;

        auto hit_rec = closest_hit(ray, params.prims.begin, params.prims.end, isect);

        if (any(hit_rec.hit))
        {
            result.hit = hit_rec.hit;
            result.isect_pos = ray.ori + ray.dir * hit_rec.t;
        }
        else
        {
            result.hit = false;
            result.color = params.bg_color;
            return result;
        }

        C color(0.0);

        size_t depth = 0;
        auto no_hit_color = C(from_rgba(params.bg_color));
        auto throughput = S(1.0);
        while (any(hit_rec.hit) && any(throughput > S(params.epsilon)) && depth++ < params.num_bounces)
        {
            hit_rec.isect_pos = ray.ori + ray.dir * hit_rec.t;

            auto surf = get_surface(hit_rec, params);
            auto ambient = surf.material.ambient() * C(from_rgba(params.ambient_color));
            auto shaded_clr = select( hit_rec.hit, ambient, C(from_rgba(params.bg_color)) );
            auto view_dir = -ray.dir;

            auto n = surf.shading_normal;

#if 1 // two-sided
            n = faceforward( n, view_dir, surf.geometric_normal );
#endif

            for (auto it = params.lights.begin; it != params.lights.end; ++it)
            {
                auto light_dir = normalize( V(it->position()) - hit_rec.isect_pos );
                R shadow_ray
                (
                    hit_rec.isect_pos + light_dir * S(params.epsilon),
                    light_dir
                );

                // only cast a shadow if occluder between light source and hit pos
                auto shadow_rec  = any_hit(
                        shadow_ray,
                        params.prims.begin,
                        params.prims.end,
                        length(hit_rec.isect_pos - V(it->position())),
                        isect
                        );

                auto active_rays = hit_rec.hit & !shadow_rec.hit;

                auto sr         = make_shade_record<Params, S>();
                sr.active       = active_rays;
                sr.isect_pos    = hit_rec.isect_pos;
                sr.normal       = n;
                sr.view_dir     = view_dir;
                sr.light_dir    = light_dir;
                sr.light        = *it;
                auto clr        = surf.shade(sr);

                shaded_clr += select( active_rays, clr, C(0.0) );
            }

            color += select( hit_rec.hit, shaded_clr, no_hit_color ) * throughput;

            auto bounce = detail::specular_bounce(surf.material, view_dir, surf.shading_normal);

            if (any(bounce.kr > S(0.0)))
            {
                auto dir = bounce.reflected_dir;
                ray = R(
                    hit_rec.isect_pos + dir * S(params.epsilon),
                    dir
                    );
                hit_rec = closest_hit(ray, params.prims.begin, params.prims.end, isect);
            }
            throughput *= bounce.kr;
            no_hit_color = C(0.0);
        }

        result.color = select( result.hit, to_rgba(color), params.bg_color );

        return result;

    }

    template <typename R>
    VSNRAY_FUNC result_record<typename R::scalar_type> operator()(R ray) const
    {
        default_intersector ignore;
        return (*this)(ignore, ray);
    }
};

} // whitted
} // visionaray

#endif // VSNRAY_WHITTED_INL
