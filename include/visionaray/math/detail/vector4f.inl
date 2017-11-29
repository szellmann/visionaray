// This file is distributed under the MIT license.
// See the LICENSE file for details.

namespace MATH_NAMESPACE
{

//-------------------------------------------------------------------------------------------------
// Specialization vector<4, float> has 16-byte alignment!
//

template <>
class VSNRAY_ALIGN(16) vector<4, float>
{   
public:

    using value_type = float; 

    float x;
    float y;
    float z;
    float w;

    vector() = default;
    MATH_FUNC vector(float x, float y, float z, float w)
        : x(x)
        , y(y)
        , z(z)
        , w(w)
    {
    }

    MATH_FUNC explicit vector(float s)
        : x(s)
        , y(s)
        , z(s)
        , w(s)
    {
    }

    MATH_FUNC explicit vector(float const data[4])
        : x(data[0])
        , y(data[1])
        , z(data[2])
        , w(data[3])
    {
    }

    template <typename U>
    MATH_FUNC explicit vector(vector<2, U> const& rhs, U const& z, U const& w)
        : x(rhs.x)
        , y(rhs.y)
        , z(z)
        , w(w)
    {
    }

    template <typename U>
    MATH_FUNC explicit vector(vector<2, U> const& first, vector<2, U> const& second)
        : x(first.x)
        , y(first.y)
        , z(second.x)
        , w(second.y)
    {
    }

    template <typename U>
    MATH_FUNC explicit vector(vector<3, U> const& rhs, U const& w)
        : x(rhs.x)
        , y(rhs.y)
        , z(rhs.z)
        , w(w)
    {
    }

    template <typename U>
    MATH_FUNC explicit vector(vector<4, U> const& rhs)
        : x(rhs.x)
        , y(rhs.y)
        , z(rhs.z)
        , w(rhs.w)
    {
    }

    template <typename U>
    MATH_FUNC vector& operator=(vector<4, U> const& rhs)
    {
        x = rhs.x;
        y = rhs.y;
        z = rhs.z;
        w = rhs.w;
        return *this;
    }

    MATH_FUNC float* data()
    {
        return reinterpret_cast<float*>(this);
    }

    MATH_FUNC float const* data() const
    {
        return reinterpret_cast<float const*>(this);
    }

    MATH_FUNC float& operator[](size_t i)
    {
        return data()[i];
    }

    MATH_FUNC float const& operator[](size_t i) const
    {
        return data()[i];
    }

    MATH_FUNC vector<2, float>& xy()
    {
        return *reinterpret_cast<vector<2, float>*>(data());
    }

    MATH_FUNC vector<2, float> const& xy() const
    {
        return *reinterpret_cast<vector<2, float> const*>(data());
    }

    MATH_FUNC vector<3, float>& xyz()
    {
        return *reinterpret_cast<vector<3, float>*>(data());
    }

    MATH_FUNC vector<3, float> const& xyz() const
    {
        return *reinterpret_cast<vector<3, float> const*>(data());
    }
};

} // MATH_NAMESPACE
