// This file is distributed under the MIT license.
// See the LICENSE file for details.

namespace visionaray
{
namespace detail
{

template <typename T>
class generic_primitive_intersection_visitor
{
public:

    using return_type = hit_record<basic_ray<T>, primitive<unsigned>>;

public:

    generic_primitive_intersection_visitor(basic_ray<T> const& ray)
        : ray_(ray)
    {
    }

    template <typename X>
    VSNRAY_FUNC
    return_type operator()(X const& ref) const
    {
        return intersect(ray_, ref);
    }

private:

    basic_ray<T> const& ray_;

};


class generic_primitive_get_bounds_visitor
{
public:

    using return_type = basic_aabb<float>;

public:

    template <typename X>
    VSNRAY_FUNC
    return_type operator()(X const& ref) const
    {
        return get_bounds(ref);
    }
};

} // detail


//-------------------------------------------------------------------------------------------------
// intersect(ray, generic_primitive)
//

template <typename T, typename ...Ts>
VSNRAY_FUNC
inline hit_record<basic_ray<T>, primitive<unsigned>> intersect(
        basic_ray<T> const&                 ray,
        generic_primitive<Ts...> const&     primitive
        )
{
    return apply_visitor( detail::generic_primitive_intersection_visitor<T>(ray), primitive );
}


//-------------------------------------------------------------------------------------------------
// Geometric functions
//

template <typename ...Ts>
VSNRAY_FUNC
auto get_bounds(generic_primitive<Ts...> const& primitive)
    -> typename detail::generic_primitive_get_bounds_visitor::return_type
{
    return apply_visitor( detail::generic_primitive_get_bounds_visitor(), primitive );
}

} // visionaray
