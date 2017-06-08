// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <visionaray/config.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <vector>

#include <visionaray/detail/parallel_algorithm.h>

#include <gtest/gtest.h>

using namespace visionaray;


#if VSNRAY_HAVE_TBB

//-------------------------------------------------------------------------------------------------
// Test counting_sort()
//

TEST(ParallelAlgorithm, CountingSort)
{
    // Array of ints
    {
        std::vector<int> a{3, 1, 4, 3, 2, 1, 8, 7, 7, 7};
        std::vector<int> b(a.size());
        std::vector<int> counts(9);

        paralgo::counting_sort(a.begin(), a.end(), b.begin(), counts);
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
        std::array<int, K> counts;

        for (size_t i = 0; i < N; ++i)
        {
            a[i] = rand() % K;
        }

        paralgo::counting_sort(a.begin(), a.end(), b.begin(), counts);
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
        std::vector<unsigned> counts(K);

        for (size_t i = 0; i < N; ++i)
        {
            a[i].key = rand() % K;
        }

        paralgo::counting_sort(
                a.begin(),
                a.end(),
                b.begin(),
                counts,
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
