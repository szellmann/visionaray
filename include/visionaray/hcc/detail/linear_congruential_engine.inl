// This file is distributed under the MIT license.
// See the LICENSE file for details.

namespace visionaray
{
namespace hcc
{

//-------------------------------------------------------------------------------------------------
// linear_congruential_engine members
//

template <typename UI, UI a, UI c, UI m>
VSNRAY_FUNC
linear_congruential_engine<UI, a, c, m>::linear_congruential_engine(UI s)
    : x_(s)
{
}

template <typename UI, UI a, UI c, UI m>
VSNRAY_FUNC
void linear_congruential_engine<UI, a, c, m>::seed(UI s)
{
    x_ = s;
}

template <typename UI, UI a, UI c, UI m>
VSNRAY_FUNC
UI linear_congruential_engine<UI, a, c, m>::operator()()
{
    // This could definitely be done more efficiently
    // by exploiting certain further knowledge about
    // types and values..
    x_ = (a * x_ + c) % m;
    return x_;
}

} // hcc
} // visionaray
