// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_MAKE_GENERATOR_H
#define VSNRAY_MAKE_GENERATOR_H 1

#include <utility>

#include "detail/macros.h"
#include "pixel_sampler_types.h"
#include "random_generator.h"

namespace visionaray
{
namespace detail
{

template <typename T, typename U>
struct make_generator_impl
{
    struct void_t
    {
        void_t() = default;

        template <typename ...Args>
        VSNRAY_FUNC void_t(Args...) {}

        T next() { return {}; }
    };

    using generator_type = void_t;
};

template <typename T>
struct make_generator_impl<T, pixel_sampler::jittered_type>
{
    using generator_type = random_generator<T>;
};

template <typename T, typename U>
struct make_generator_impl<T, pixel_sampler::basic_jittered_blend_type<U>>
{
    using generator_type = random_generator<T>;
};

} // detail


//-------------------------------------------------------------------------------------------------
// Factory function for number generators
//

template <typename T, typename PixelSampler, typename ...Args>
VSNRAY_FUNC
auto make_generator(T /* */, PixelSampler /* */, Args&&... args)
    -> typename detail::make_generator_impl<T, PixelSampler>::generator_type
{
    return typename detail::make_generator_impl<T, PixelSampler>::generator_type(
            std::forward<Args>(args)...
            );
}

} // visionaray

#endif // VSNRAY_MAKE_GENERATOR_H
