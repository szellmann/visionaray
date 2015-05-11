// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <sstream>
#include <string>

#include <visionaray/math/math.h>
#include <visionaray/variant.h>

#include <gtest/gtest.h>

using namespace visionaray;


//-------------------------------------------------------------------------------------------------
// Test entities
//

struct int_visitor
{
    using return_type = int;

    int operator()(int i) const
    {
        return i;
    }
};

struct is_double_visitor
{
    using return_type = bool;

    template <typename X>
    bool operator()(X) const
    {
        return false;
    }

    bool operator()(double) const
    {
        return true;
    }
};


template <typename T>
struct some_struct
{
    T t;
    vector<3, T> u;
    vector<4, T> v;
};


struct some_struct_visitor
{
    using return_type = std::string;

    template <typename X>
    std::string operator()(X const& x) const
    {
        std::stringstream stream;
        stream <<  "t: " << x.t;
        return stream.str();
    }
};


//-------------------------------------------------------------------------------------------------
// Unit tests
//

TEST(VariantTest, General)
{
    // int

    variant<int> var_i = 4711;
    EXPECT_EQ( *var_i.as<int>(), 4711 );
    EXPECT_EQ( apply_visitor( int_visitor(), var_i ), 4711 );


    // vec3

    variant<vec3> var_v3 = vec3{ 1.0f, 2.0f, 3.0f };
    auto v = *var_v3.as<vec3>();
    EXPECT_FLOAT_EQ( v.x, 1.0f );
    EXPECT_FLOAT_EQ( v.y, 2.0f );
    EXPECT_FLOAT_EQ( v.z, 3.0f );


    // test for a certain type

    variant<int, double> var_id1 = double(0.0);
    EXPECT_TRUE( apply_visitor( is_double_visitor(), var_id1 ) );


    // struct with some members

    some_struct<double> sstruct;
    sstruct.t = 23.0;
    variant<some_struct<float>, some_struct<double>> var_struct = sstruct;
    auto str1 = apply_visitor( some_struct_visitor(), var_struct );
    EXPECT_FALSE( str1.empty() );


    // copy constructor

    auto var_copy = var_struct;
    auto str2 = apply_visitor( some_struct_visitor(), var_copy );
    EXPECT_FALSE( str1.empty() );

    EXPECT_STREQ( str1.c_str(), str2.c_str() );
}


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::FLAGS_gtest_death_test_style = "fast";
    return RUN_ALL_TESTS();
}
