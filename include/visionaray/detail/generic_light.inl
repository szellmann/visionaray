// This file is distributed under the MIT license.
// See the LICENSE file for details.

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Public interface
//

template <typename T, typename ...Ts>
template <typename L>
generic_light<T, Ts...>::generic_light(L const& light)
    : generic_light<T, Ts...>::base_type(light)
{
}

template <typename T, typename ...Ts>
template <typename U>
VSNRAY_FUNC
vector<3, U> generic_light<T, Ts...>::intensity(vector<3, U> const& pos) const
{
    return apply_visitor( intensity_visitor<U>(pos), *this );
}

template <typename T, typename ...Ts>
template <typename Generator, typename U>
VSNRAY_FUNC
light_sample<U> generic_light<T, Ts...>::sample(Generator& gen) const
{
    return apply_visitor( sample_visitor<Generator, U>(gen), *this );
}

template <typename T, typename ...Ts>
VSNRAY_FUNC
vector<3, typename T::scalar_type> generic_light<T, Ts...>::position() const
{
    return apply_visitor( position_visitor(), *this );
}


//-------------------------------------------------------------------------------------------------
// Private variant visitors
//

template <typename T, typename ...Ts>
template <typename U>
struct generic_light<T, Ts...>::intensity_visitor
{
    using Base = generic_light<T, Ts...>;
    using return_type = vector<3, U>;

    VSNRAY_FUNC
    intensity_visitor(vector<3, U> const& pos) : pos_(pos) {}

    template <typename X>
    VSNRAY_FUNC
    return_type operator()(X const& ref) const
    {
        return ref.intensity(pos_);
    }

    vector<3, U> const& pos_;
};

template <typename T, typename ...Ts>
template <typename Generator, typename U>
struct generic_light<T, Ts...>::sample_visitor
{
    using Base = generic_light<T, Ts...>;
    using return_type = light_sample<U>;

    VSNRAY_FUNC
    sample_visitor(Generator& gen) : gen_(gen) {}

    template <typename X>
    VSNRAY_FUNC
    return_type operator()(X const& ref) const
    {
        return ref.sample(gen_);
    }

    Generator& gen_;
};

template <typename T, typename ...Ts>
struct generic_light<T, Ts...>::position_visitor
{
    using Base = generic_light<T, Ts...>;
    using return_type = vector<3, typename T::scalar_type>;

    template <typename X>
    VSNRAY_FUNC
    return_type operator()(X const& ref) const
    {
        return ref.position();
    }
};

} // visionaray
