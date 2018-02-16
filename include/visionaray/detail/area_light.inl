// This file is distributed under the MIT license.
// See the LICENSE file for details.

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// area_light members
//

template <typename Geometry>
inline area_light<Geometry>::area_light(Geometry geometry)
    : geometry_(geometry)
{
}

template <typename Geometry>
template <typename T>
VSNRAY_FUNC
inline vector<3, T> area_light<Geometry>::intensity(vector<3, T> const& pos) const
{
}

template <typename Geometry>
template <typename T, typename Sampler>
VSNRAY_FUNC
inline vector<3, T> area_light<Geometry>::sample(T& pdf, Sampler& samp) const
{
    return sample_surface(geometry_, pdf, samp);
}

template <typename Geometry>
template <typename T, size_t N, typename Sampler>
VSNRAY_FUNC
inline void area_light<Geometry>::sample(
        array<T, N>& pdfs,
        array<vector<3, T>, N>& result,
        Sampler& samp
        ) const
{
    for (size_t i = 0; i < N; ++i)
    {
        result[i] = sample(pdfs[i], samp);
    }
}

template <typename Geometry>
VSNRAY_FUNC
inline vector<3, float> area_light<Geometry>::position() const
{
    return get_bounds(geometry_).center();
}

} // visionaray
