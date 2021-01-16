// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_DETAIL_WHITTED_INL
#define VSNRAY_DETAIL_WHITTED_INL 1

#include <type_traits>

#include <visionaray/array.h>
#include <visionaray/generic_material.h>
#include <visionaray/get_surface.h>
#include <visionaray/result_record.h>
#include <visionaray/spectrum.h>
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
        mat.kr()
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

template <
    unsigned N,
    typename ...Ts,
    typename T,
    typename = typename std::enable_if<simd::is_simd_vector<T>::value>::type
    >
inline auto specular_bounce(
        simd::generic_material<N, Ts...> const& mat,
        vector<3, T> const&                     view_dir,
        vector<3, T> const&                     normal
        )
    -> bounce_result<vector<3, T>, T>
{
    using float_array = simd::aligned_array_t<T>;

    auto ms  = unpack(mat);
    auto vds = unpack(view_dir);
    auto ns  = unpack(normal);

    array<vector<3, float>, N> refl_dir;
    array<vector<3, float>, N> refr_dir;
    float_array                kr;
    float_array                kt;

    for (unsigned i = 0; i < N; ++i)
    {
        auto res = specular_bounce(ms[i], vds[i], ns[i]);
        refl_dir[i] = res.reflected_dir;
        refr_dir[i] = res.refracted_dir;
        kr[i]       = res.kr;
        kt[i]       = res.kt;
    }

    return make_bounce_result(
        simd::pack(refl_dir),
        simd::pack(refr_dir),
        T(kr),
        T(kt)
        );
}

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
            result.color = vector<4, S>(params.environment_map.background_intensity(ray.dir), S(1.0));
            return result;
        }

        C color(0.0);

        unsigned depth = 0;
        C no_hit_color(from_rgb(params.environment_map.background_intensity(ray.dir)));
        S throughput(1.0);
        while (any(hit_rec.hit) && any(throughput > S(params.epsilon)) && depth++ < params.num_bounces)
        {
            hit_rec.isect_pos = ray.ori + ray.dir * hit_rec.t;

            auto surf = get_surface(hit_rec, params);
            auto env = params.environment_map.intensity(ray.dir);
            auto bgcolor = params.environment_map.background_intensity(ray.dir);
            auto ambient = surf.material.ambient() * C(from_rgb(env));
            auto shaded_clr = select( hit_rec.hit, ambient, C(from_rgb(bgcolor)) );
            auto view_dir = -ray.dir;

            for (auto it = params.lights.begin; it != params.lights.end; ++it)
            {
                auto light_dir = normalize( V(it->position()) - hit_rec.isect_pos );

                auto clr = surf.shade(view_dir, light_dir, it->intensity(hit_rec.isect_pos));

                R shadow_ray(
                        hit_rec.isect_pos + light_dir * S(params.epsilon),
                        light_dir
                        );

                // only cast a shadow if occluder between light source and hit pos
                auto shadow_rec = any_hit(
                        shadow_ray,
                        params.prims.begin,
                        params.prims.end,
                        length(hit_rec.isect_pos - V(it->position())),
                        isect
                        );

                shaded_clr += select(
                        hit_rec.hit & !shadow_rec.hit,
                        clr,
                        C(0.0)
                        );
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

        result.color = select(
                result.hit,
                to_rgba(color),
                vector<4, S>(params.environment_map.intensity(ray.dir), S(1.0))
                );

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

#endif // VSNRAY_DETAIL_WHITTED_INL
