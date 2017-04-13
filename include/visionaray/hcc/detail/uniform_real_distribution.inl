// This file is distributed under the MIT license.
// See the LICENSE file for details.

// This file contains source code from the thrust library
// Original license follows (Apache 2.0)


//
//  Copyright 2008-2013 NVIDIA Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//

namespace visionaray
{
namespace hcc
{

//-------------------------------------------------------------------------------------------------
// uniform_real_distribution members
//

template <typename T>
VSNRAY_FUNC
uniform_real_distribution<T>::uniform_real_distribution(T a, T b)
    : param_(a, b)
{
}

template <typename T>
VSNRAY_FUNC
uniform_real_distribution<T>::uniform_real_distribution(typename uniform_real_distribution<T>::param_type const& param)
    : param_(param)
{
}

template <typename T>
template <typename URNG>
VSNRAY_FUNC
typename uniform_real_distribution<T>::result_type uniform_real_distribution<T>::operator()(URNG& urng)
{
    return operator()(urng, param_);
}

template <typename T>
template <typename URNG>
VSNRAY_FUNC
typename uniform_real_distribution<T>::result_type uniform_real_distribution<T>::operator()(
        URNG& urng,
        typename uniform_real_distribution<T>::param_type const& param
        )
{
    // call the urng & map its result to [0,1)
    result_type result = static_cast<result_type>(urng() - URNG::min);

    // adding one to the denominator ensures that the interval is half-open at 1.0
    // XXX adding 1.0 to a potentially large floating point number seems like a bad idea
    // XXX OTOH adding 1 to what is potentially UINT_MAX also seems like a bad idea
    // XXX we could statically check if 1u + (max - min) is representable and do that, otherwise use the current implementation
    result /= result_type(1) + static_cast<result_type>(URNG::max - URNG::min);

    return (result * (param.second - param.first)) + param.first;
}

template <typename T>
VSNRAY_FUNC
typename uniform_real_distribution<T>::result_type uniform_real_distribution<T>::a() const
{
    return param_.first;
}

template <typename T>
VSNRAY_FUNC
typename uniform_real_distribution<T>::result_type uniform_real_distribution<T>::b() const
{
    return param_.second;
}

template <typename T>
VSNRAY_FUNC
typename uniform_real_distribution<T>::param_type uniform_real_distribution<T>::param() const
{
    return param_;
}

template <typename T>
VSNRAY_FUNC
void uniform_real_distribution<T>::param(typename uniform_real_distribution<T>::param_type const& param) const
{
    param_ = param;
}

template <typename T>
VSNRAY_FUNC
typename uniform_real_distribution<T>::result_type uniform_real_distribution<T>::min() const
{
    return param_.first;
}

template <typename T>
VSNRAY_FUNC
typename uniform_real_distribution<T>::result_type uniform_real_distribution<T>::max() const
{
    return param_.second;
}

} // hcc
} // visionaray
