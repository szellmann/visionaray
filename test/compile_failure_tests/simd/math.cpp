#include "visionaray/math/math.h"

using namespace visionaray;


int main()
{

#if defined MATH_DIV_UP

    int a = div_up(1,2);

#elif defined MATH_ROUND_UP

    int a = round_up(1,2);

#elif defined MATH_DIV_UP_FLOAT

    auto a = div_up(1.f,2.f);

#elif defined MATH_ROUND_UP_FLOAT

    auto a = round_up(1.f,2.f);

#endif

    return 0;
}
