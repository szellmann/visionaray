// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <algorithm>
#include <cstddef>
#include <cstring> // memcpy

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>
#include <thrust/swap.h>

#include <visionaray/math/array.h>

#include <gtest/gtest.h>

using namespace visionaray;


//-------------------------------------------------------------------------------------------------
// Test reverse iterators
//

template <typename FwdIt, typename T>
__device__ void iota(FwdIt first, FwdIt last, T value)
{
    for (auto it = first; it != last; ++it)
    {
        *it = value++;
    }
}

template <typename Array>
__global__ void kernel_reverse_it(bool* mem, Array /* */)
{
    Array arr1;
    iota(arr1.begin(), arr1.end(), 0);

    // Test non-const iterators for writing
    Array arr2;
    iota(arr2.rbegin(), arr2.rend(), 0);


    size_t i = 0;

    // Test const reverse iterators obtained implicitly through rbegin() and rend()
    auto it1 = arr1.rbegin();
    auto it2 = arr2.begin();
    for (; it1 != arr1.rend() && it2 != arr2.end(); ++it1, ++it2)
    {
        mem[i++] = *it1 == *it2;
    }

    // Test const reverse iterators obtained through crbegin() and crend()
    auto cit1 = arr1.crbegin();
    auto cit2 = arr2.cbegin();
    for (; cit1 != arr1.crend() && cit2 != arr2.cend(); ++cit1, ++cit2)
    {
        mem[i++] = *cit1 == *cit2;
    }
}

TEST(ArrayCU, ReverseIt)
{
    static const size_t N = 50;

    thrust::device_vector<bool> d_result(N * 2);
    thrust::fill(d_result.begin(), d_result.end(), false);

    kernel_reverse_it<<<1, 1>>>(
            thrust::raw_pointer_cast(d_result.data()),
            array<int, N>{}
            );

    thrust::host_vector<bool> h_result(d_result);

    for (auto b : h_result)
    {
        EXPECT_TRUE(b);
    }
}


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


//-------------------------------------------------------------------------------------------------
// Test interoperability with thrust::swap()
//

template <typename T, typename Array>
__global__ void kernel_thrust_swap(T* mem, Array /* */)
{
    Array arr1;
    Array arr2;

    memcpy(arr1.data(), mem, sizeof(arr1));
    memcpy(arr2.data(), mem + arr1.size(), sizeof(arr2));

    thrust::swap(arr1, arr2);

    memcpy(mem, arr1.data(), sizeof(arr1));
    memcpy(mem + arr1.size(), arr2.data(), sizeof(arr2));
}

TEST(ArrayCU, ThrustSwap)
{
    static const size_t N = 50;

    thrust::host_vector<int> h_data(N * 2);
    std::fill(h_data.data(), h_data.data() + N, 23);
    std::fill(h_data.data() + N, h_data.data() + h_data.size(), 24);
    thrust::device_vector<int> d_data(h_data);

    kernel_thrust_swap<<<1, 1>>>(
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


//-------------------------------------------------------------------------------------------------
// Test comparisons
//

__global__ void kernel_compare(bool* result)
{
    const int N = 50;

    array<int, N> arr1;
    arr1.fill(23);

    array<int, N> arr2;
    arr2.fill(24);

    array<int, N> arr3;
    arr3.fill(23);

    array<int, N> arr4;
    array<int, N> arr5;
    for (int i = 0; i < N; ++i)
    {
        arr4[i] = i;
        arr5[i] = (i + 1) % N;
    }

    result[0] = ( arr1 == arr1 );
    result[1] = ( arr1 != arr2 );
    result[2] = ( arr2 != arr1 );
    result[3] = ( arr1 == arr3 );
    result[4] = ( arr3 == arr1 );
    result[5] = ( arr4 != arr5 );
    result[6] = ( arr5 != arr4 );
}


TEST(ArrayCU, Compare)
{
    thrust::device_vector<bool> d_result(7);
    thrust::fill(d_result.begin(), d_result.end(), false);

    kernel_compare<<<1, 1>>>(thrust::raw_pointer_cast(d_result.data()));

    thrust::host_vector<bool> h_result(d_result);

    for (auto b : h_result)
    {
        EXPECT_TRUE(b);
    }
}


//-------------------------------------------------------------------------------------------------
// Test element access with thrust::get()
//

__global__ void kernel_get(bool* mem)
{
    array<int, 3> arr;
    thrust::get<0>(arr) = 0;
    thrust::get<1>(arr) = 1;
    thrust::get<2>(arr) = 2;

    mem[0] = arr[0] == 0;
    mem[1] = arr[1] == 1;
    mem[2] = arr[2] == 2;

    mem[3] = thrust::get<0>(arr) == 0;
    mem[4] = thrust::get<1>(arr) == 1;
    mem[5] = thrust::get<2>(arr) == 2;
}

TEST(ArrayCU, Get)
{
    thrust::device_vector<bool> d_result(6);
    thrust::fill(d_result.begin(), d_result.end(), false);

    kernel_get<<<1, 1>>>(thrust::raw_pointer_cast(d_result.data()));

    thrust::host_vector<bool> h_result(d_result);

    for (auto b : h_result)
    {
        EXPECT_TRUE(b);
    }
}
