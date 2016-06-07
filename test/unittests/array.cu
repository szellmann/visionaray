// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <algorithm>
#include <cstddef>
#include <cstring> // memcpy

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <visionaray/array.h>

#include <gtest/gtest.h>

using namespace visionaray;


//-------------------------------------------------------------------------------------------------
// Test array::fill()
//

template <typename T, typename Array>
__global__ void kernel_fill(T value, T* mem, Array /* */)
{
    Array arr;
    arr.fill(value);

    // Copy to global memory so we can compare on the host
    memcpy(mem, arr.data(), sizeof(arr));
}

TEST(ArrayCU, Fill)
{
    static const size_t N = 50;
    thrust::device_vector<int> d_result(N);
    int value = 23;

    kernel_fill<<<1, 1>>>(
            value,
            thrust::raw_pointer_cast(d_result.data()),
            array<int, N>{}
            );

    thrust::host_vector<int> h_result(d_result);

    for (size_t i = 0; i < N; ++i)
    {
        EXPECT_EQ(h_result[i], value);
    }
}


//-------------------------------------------------------------------------------------------------
// Test array::swap()
//

template <typename T, typename Array>
__global__ void kernel_swap(T* mem, Array /* */)
{
    Array arr1;
    Array arr2;

    memcpy(arr1.data(), mem, sizeof(arr1));
    memcpy(arr2.data(), mem + arr1.size(), sizeof(arr2));

    arr1.swap(arr2);

    memcpy(mem, arr1.data(), sizeof(arr1));
    memcpy(mem + arr1.size(), arr2.data(), sizeof(arr2));
}

TEST(ArrayCU, Swap)
{
    static const size_t N = 50;

    thrust::host_vector<int> h_data(N * 2);
    std::fill(h_data.data(), h_data.data() + N, 23);
    std::fill(h_data.data() + N, h_data.data() + h_data.size(), 24);
    thrust::device_vector<int> d_data(h_data);

    kernel_swap<<<1, 1>>>(
            thrust::raw_pointer_cast(d_data.data()),
            array<int, N>{}
            );

    thrust::copy(d_data.begin(), d_data.end(), h_data.begin());

    for (size_t i = 0; i < N; ++i)
    {
        EXPECT_EQ(h_data[i], 24);
    }

    for (size_t i = N; i < N * 2; ++i)
    {
        EXPECT_EQ(h_data[i], 23);
    }

}
