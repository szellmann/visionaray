// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <visionaray/config.h>

#include <algorithm>
#include <array>
#include <climits>
#include <cstdlib>
#include <functional>
#include <vector>

#include <visionaray/detail/algorithm.h>

#include <gtest/gtest.h>

using namespace visionaray;


//-------------------------------------------------------------------------------------------------
// Test counting_sort()
//

TEST(Algorithm, CountingSort)
{
    // Array of ints
    {
        std::vector<int> a{3, 1, 4, 3, 2, 1, 8, 7, 7, 7};
        std::vector<int> b(a.size());

        algo::counting_sort<9>(a.begin(), a.end(), b.begin());
        EXPECT_TRUE(std::is_sorted(b.begin(), b.end()));

        std::sort(a.begin(), a.end());
        EXPECT_TRUE(a == b);
    }

    // Larger array of ints
    {
        static const size_t N = 1000000;
        static const size_t K = 256;

        std::vector<int> a(N);
        std::vector<int> b(N);

        for (size_t i = 0; i < N; ++i)
        {
            a[i] = rand() % K;
        }

        algo::counting_sort<K>(a.begin(), a.end(), b.begin());
        EXPECT_TRUE(std::is_sorted(b.begin(), b.end()));

        std::sort(a.begin(), a.end());
        EXPECT_TRUE(a == b);
    }

    // User defined struct
    {
        struct my_struct
        {
            float values[30];
            long key;

            bool operator==(my_struct const& rhs) const
            {
                for (int i = 0; i < 30; ++i)
                {
                    if (values[i] != rhs.values[i])
                    {
                        return false;
                    }
                }

                return key == rhs.key;
            }
        };


        static const size_t N = 10000;
        static const size_t K = 20;

        std::vector<my_struct> a(N);
        std::vector<my_struct> b(N);

        for (size_t i = 0; i < N; ++i)
        {
            a[i].key = rand() % K;
        }

        algo::counting_sort<K>(
                a.begin(),
                a.end(),
                b.begin(),
                [](my_struct const& val) { return val.key; }
                );
        EXPECT_TRUE( std::is_sorted(
                b.begin(),
                b.end(),
                [](my_struct const& val1, my_struct const& val2) { return val1.key < val2.key; }
                ) );

        std::sort(
                a.begin(),
                a.end(),
                [](my_struct const& val1, my_struct const& val2) { return val1.key < val2.key; }
                );
        EXPECT_TRUE(a == b);
    }
}


#if VSNRAY_HAVE_TBB

//-------------------------------------------------------------------------------------------------
// Test parallel_counting_sort()
//

TEST(Algorithm, ParallelCountingSort)
{
    // Array of ints
    {
        std::vector<int> a{3, 1, 4, 3, 2, 1, 8, 7, 7, 7};
        std::vector<int> b(a.size());

        algo::parallel_counting_sort<9>(a.begin(), a.end(), b.begin());
        EXPECT_TRUE(std::is_sorted(b.begin(), b.end()));

        std::sort(a.begin(), a.end());
        EXPECT_TRUE(a == b);
    }

    // Larger array of ints
    {
        static const size_t N = 1000000;
        static const size_t K = 256;

        std::vector<int> a(N);
        std::vector<int> b(N);

        for (size_t i = 0; i < N; ++i)
        {
            a[i] = rand() % K;
        }

        algo::parallel_counting_sort<K>(a.begin(), a.end(), b.begin());
        EXPECT_TRUE(std::is_sorted(b.begin(), b.end()));

        std::sort(a.begin(), a.end());
        EXPECT_TRUE(a == b);
    }

    // User defined struct
    {
        struct my_struct
        {
            float values[30];
            long key;

            bool operator==(my_struct const& rhs) const
            {
                for (int i = 0; i < 30; ++i)
                {
                    if (values[i] != rhs.values[i])
                    {
                        return false;
                    }
                }

                return key == rhs.key;
            }
        };


        static const size_t N = 10000;
        static const size_t K = 20;

        std::vector<my_struct> a(N);
        std::vector<my_struct> b(N);

        for (size_t i = 0; i < N; ++i)
        {
            a[i].key = rand() % K;
        }

        algo::parallel_counting_sort<K>(
                a.begin(),
                a.end(),
                b.begin(),
                [](my_struct const& val) { return val.key; }
                );
        EXPECT_TRUE( std::is_sorted(
                b.begin(),
                b.end(),
                [](my_struct const& val1, my_struct const& val2) { return val1.key < val2.key; }
                ) );

        std::sort(
                a.begin(),
                a.end(),
                [](my_struct const& val1, my_struct const& val2) { return val1.key < val2.key; }
                );
        EXPECT_TRUE(a == b);
    }
}

#endif // VSNRAY_HAVE_TBB


//-------------------------------------------------------------------------------------------------
// Test insert_sorted()
//

TEST(Algorithm, InsertSorted)
{
    // Array of ints

    std::array<int, 4> arr = {{ INT_MAX, INT_MAX, INT_MAX, INT_MAX }};

    algo::insert_sorted(0, arr.begin(), arr.end(), std::less<int>());
    // 0, inf, inf, inf
    EXPECT_EQ(arr[0], 0);
    EXPECT_EQ(arr[1], INT_MAX);
    EXPECT_EQ(arr[2], INT_MAX);
    EXPECT_EQ(arr[3], INT_MAX);

    algo::insert_sorted(10, arr.begin(), arr.end(), std::less<int>());
    // 0, 10, inf, inf
    EXPECT_EQ(arr[0], 0);
    EXPECT_EQ(arr[1], 10);
    EXPECT_EQ(arr[2], INT_MAX);
    EXPECT_EQ(arr[3], INT_MAX);

    algo::insert_sorted(4, arr.begin(), arr.end(), std::less<int>());
    // 0, 4, 10, inf
    EXPECT_EQ(arr[0], 0);
    EXPECT_EQ(arr[1], 4);
    EXPECT_EQ(arr[2], 10);
    EXPECT_EQ(arr[3], INT_MAX);

    algo::insert_sorted(4, arr.begin(), arr.end(), std::less<int>());
    // 0, 4, 4, 10
    EXPECT_EQ(arr[0], 0);
    EXPECT_EQ(arr[1], 4);
    EXPECT_EQ(arr[2], 4);
    EXPECT_EQ(arr[3], 10);

    algo::insert_sorted(4, arr.begin(), arr.end(), std::less<int>());
    // 0, 4, 4, 4
    EXPECT_EQ(arr[0], 0);
    EXPECT_EQ(arr[1], 4);
    EXPECT_EQ(arr[2], 4);
    EXPECT_EQ(arr[3], 4);

    algo::insert_sorted(10, arr.begin(), arr.end(), std::less<int>());
    // 0, 4, 4, 4 (10 is not inserted!)
    EXPECT_EQ(arr[0], 0);
    EXPECT_EQ(arr[1], 4);
    EXPECT_EQ(arr[2], 4);
    EXPECT_EQ(arr[3], 4);

    algo::insert_sorted(-INT_MAX, arr.begin(), arr.end(), std::less<int>());
    // -inf, 0, 4, 4
    EXPECT_EQ(arr[0], -INT_MAX);
    EXPECT_EQ(arr[1], 0);
    EXPECT_EQ(arr[2], 4);
    EXPECT_EQ(arr[3], 4);
}
