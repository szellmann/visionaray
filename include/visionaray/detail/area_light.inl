// This file is distributed under the MIT license.
// See the LICENSE file for details.

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// area_light members
//

template <typename T, typename Geometry>
inline area_light<T, Geometry>::area_light(Geometry geometry)
    : geometry_(geometry)
{
}

template <typename T, typename Geometry>
VSNRAY_FUNC
inline Geometry& area_light<T, Geometry>::geometry()
{
    return geometry_;
}

template <typename T, typename Geometry>
VSNRAY_FUNC
inline Geometry const& area_light<T, Geometry>::geometry() const
{
    return geometry_;
}

template <typename T, typename Geometry>
template <typename U>
VSNRAY_FUNC
inline vector<3, U> area_light<T, Geometry>::intensity(vector<3, U> const& pos) const
{
}

template <typename T, typename Geometry>
template <typename U, typename Generator>
VSNRAY_FUNC
inline vector<3, U> area_light<T, Geometry>::sample(U& pdf, Generator& gen) const
{
    return sample_surface(geometry_, pdf, gen);
}

template <typename T, typename Geometry>
template <size_t N, typename U, typename Generator>
VSNRAY_FUNC
inline void area_light<T, Geometry>::sample(
        array<U, N>& pdfs,
        array<vector<3, U>, N>& result,
        Generator& gen
        ) const
{
    for (size_t i = 0; i < N; ++i)
    {
        result[i] = sample(pdfs[i], gen);
    }
}

template <typename T, typename Geometry>
VSNRAY_FUNC
inline vector<3, T> area_light<T, Geometry>::position() const
{
    return vector<3, T>(get_bounds(geometry_).center());
}

} // visionaray
