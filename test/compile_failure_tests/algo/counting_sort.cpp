// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cstddef>
#include <cstdlib>
#include <vector>

#include <visionaray/detail/algorithm.h>
#include <visionaray/math/array.h>

using namespace visionaray;


int main()
{

#if defined(SORT_INTEGRAL_KEYS_SIMPLE)

    std::vector<int> a{3, 1, 4, 3, 2, 1, 8, 7, 7, 7};
    std::vector<int> b(a.size());
    array<unsigned, 9> counts;
    algo::counting_sort(a.begin(), a.end(), b.begin(), counts);

#elif defined(SORT_FLOATING_KEYS_SIMPLE)

    // floating-point sequences, should fail!
    std::vector<float> a{3.0f, 1.0f, 4.0f, 3.0f, 2.0f, 1.0f, 8.0f, 7.0f, 7.0f, 7.0f};
    std::vector<float> b(a.size());
    array<unsigned, 9> counts;
    algo::counting_sort(a.begin(), a.end(), b.begin(), counts);

#elif defined(SORT_INTEGRAL_KEYS_CUSTOM_TYPE)

    struct my_struct
    {
        long key;
        float value[32];
    };

    static const size_t N = 1000;
    static const size_t K = 10;

    std::vector<my_struct> a(N);
    std::vector<my_struct> b(N);
    array<unsigned, K> counts;

    for (auto& aa : a)
    {
        aa.key = rand() % K;
    }

    algo::counting_sort(
            a.begin(),
            a.end(),
            b.begin(),
            counts,
            [](my_struct const& val) { return val.key; }
            );

#elif defined(SORT_FLOATING_KEYS_CUSTOM_TYPE)

    struct my_struct
    {
        float key; // floating-point key, should fail!
        float value[32];
    };

    static const size_t N = 1000;
    static const size_t K = 10;

    std::vector<my_struct> a(N);
    std::vector<my_struct> b(N);
    array<unsigned, K> counts;

    for (auto& aa : a)
    {
        aa.key = static_cast<float>(rand() % K);
    }

    algo::counting_sort(
            a.begin(),
            a.end(),
            b.begin(),
            counts,
            [](my_struct const& val) { return val.key; }
            );

#endif

    return 0;
}
