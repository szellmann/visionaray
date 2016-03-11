// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <visionaray/math/math.h>

using namespace visionaray;


int main()
{

#if defined MATH_DIV_UP

    int a = div_up(1, 2);

#elif defined MATH_ROUND_UP

    int a = round_up(1, 2);

#elif defined MATH_DIV_UP_FLOAT

    auto a = div_up(1.0f, 2.0f);

#elif defined MATH_ROUND_UP_FLOAT

    auto a = round_up(1.0f, 2.0f);

#endif

    return 0;
}
