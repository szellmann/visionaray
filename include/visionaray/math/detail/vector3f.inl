// This file is distributed under the MIT license.
// See the LICENSE file for details.

namespace MATH_NAMESPACE
{

//-------------------------------------------------------------------------------------------------
// Specialization vector<3, float> has 16-byte alignment!
//

template <>
class VSNRAY_ALIGN(16) vector<3, float>
{   
public:

    using value_type = float; 

    float x;
    float y;
    float z;

    vector() = default;
    MATH_FUNC vector(float x, float y, float z)
        : x(x)
        , y(y)
        , z(z)
    {
    }

    MATH_FUNC explicit vector(float s)
        : x(s)
        , y(s)
        , z(s)
    {
    }

    MATH_FUNC explicit vector(float const data[3])
        : x(data[0])
        , y(data[1])
        , z(data[2])
    {
    }

    template <typename U>
    MATH_FUNC explicit vector(vector<2, U> const& rhs, U const& z)
        : x(rhs.x)
        , y(rhs.y)
        , z(z)
    {
    }

    template <typename U>
    MATH_FUNC explicit vector(vector<3, U> const& rhs)
        : x(rhs.x)
        , y(rhs.y)
        , z(rhs.z)
    {
    }

    template <typename U>
    MATH_FUNC explicit vector(vector<4, U> const& rhs)
        : x(rhs.x)
        , y(rhs.y)
        , z(rhs.z)
    {
    }

    template <typename U>
    MATH_FUNC vector& operator=(vector<3, U> const& rhs)
    {
        x = rhs.x;
        y = rhs.y;
        z = rhs.z;
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
};

} // MATH_NAMESPACE
