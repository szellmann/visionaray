// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include "Support/StringSplit.h"

#include <vector>
#include <iostream>

#include <gtest/gtest.h>

using namespace support;
using namespace support::strings;

template <class T> using IsStringRef = typename std::is_same<T, StringRef>::type;
template <class T> using IsStdString = typename std::is_same<T, std::string>::type;

using strings::details::Split_string_t;

static_assert( IsStringRef< Split_string_t<char*                >>::value, "" );
static_assert( IsStringRef< Split_string_t<char*&               >>::value, "" );
static_assert( IsStringRef< Split_string_t<char*&&              >>::value, "" );
static_assert( IsStringRef< Split_string_t<char (&)[1]          >>::value, "" );
static_assert( IsStringRef< Split_string_t<char const*          >>::value, "" );
static_assert( IsStringRef< Split_string_t<char const*&         >>::value, "" );
static_assert( IsStringRef< Split_string_t<char const*&&        >>::value, "" );
static_assert( IsStringRef< Split_string_t<char const (&)[1]    >>::value, "" );
static_assert( IsStringRef< Split_string_t<StringRef            >>::value, "" );
static_assert( IsStringRef< Split_string_t<StringRef&           >>::value, "" );
static_assert( IsStringRef< Split_string_t<StringRef&&          >>::value, "" );
static_assert( IsStringRef< Split_string_t<StringRef const      >>::value, "" );
static_assert( IsStringRef< Split_string_t<StringRef const&     >>::value, "" );
static_assert( IsStringRef< Split_string_t<StringRef const&&    >>::value, "" );
static_assert( IsStringRef< Split_string_t<std::string&         >>::value, "" );
static_assert( IsStringRef< Split_string_t<std::string const&   >>::value, "" );

static_assert( IsStdString< Split_string_t<std::string          >>::value, "" );
static_assert( IsStdString< Split_string_t<std::string&&        >>::value, "" );
static_assert( IsStdString< Split_string_t<std::string const    >>::value, "" );
static_assert( IsStdString< Split_string_t<std::string const&&  >>::value, "" );

TEST(Test, EmptyStrings)
{
    {
        auto vec = std::vector<StringRef>(split(StringRef(), ","));

        ASSERT_EQ(1, vec.size());
        EXPECT_EQ("", vec[0]);
    }
    {
        auto vec = std::vector<StringRef>(split("", ","));

        ASSERT_EQ(1, vec.size());
        EXPECT_EQ("", vec[0]);
    }
    {
        auto vec = std::vector<StringRef>(split(StringRef(), AnyOfDelimiter(",")));

        ASSERT_EQ(1, vec.size());
        EXPECT_EQ("", vec[0]);
    }
    {
        auto vec = std::vector<StringRef>(split("", AnyOfDelimiter(",")));

        ASSERT_EQ(1, vec.size());
        EXPECT_EQ("", vec[0]);
    }
}

TEST(Test, EmptyDelimiters)
{
    {
        auto vec = std::vector<StringRef>(split(StringRef(), ""));

        ASSERT_EQ(1, vec.size());
        EXPECT_EQ("", vec[0]);
    }
    {
        auto vec = std::vector<StringRef>(split(StringRef(), AnyOfDelimiter("")));

        ASSERT_EQ(1, vec.size());
        EXPECT_EQ("", vec[0]);
    }
    {
        auto vec = std::vector<StringRef>(split("", ""));

        ASSERT_EQ(1, vec.size());
        EXPECT_EQ("", vec[0]);
    }
    {
        auto vec = std::vector<StringRef>(split("", AnyOfDelimiter("")));

        ASSERT_EQ(1, vec.size());
        EXPECT_EQ("", vec[0]);
    }
    {
        auto vec = std::vector<StringRef>(split("x", ""));

        ASSERT_EQ(1, vec.size());
        EXPECT_EQ("x", vec[0]);
    }
    {
        auto vec = std::vector<StringRef>(split("x", AnyOfDelimiter("")));

        ASSERT_EQ(1, vec.size());
        EXPECT_EQ("x", vec[0]);
    }
    {
        auto vec = std::vector<StringRef>(split("abc", ""));

#if SUPPORT_STD_SPLIT
        ASSERT_EQ(3, vec.size());
        EXPECT_EQ("a", vec[0]);
        EXPECT_EQ("b", vec[1]);
        EXPECT_EQ("c", vec[2]);
#else
        ASSERT_EQ(1, vec.size());
        EXPECT_EQ("abc", vec[0]);
#endif
    }
    {
        auto vec = std::vector<StringRef>(split("abc", AnyOfDelimiter("")));

#if SUPPORT_STD_SPLIT
        ASSERT_EQ(3, vec.size());
        EXPECT_EQ("a", vec[0]);
        EXPECT_EQ("b", vec[1]);
        EXPECT_EQ("c", vec[2]);
#else
        ASSERT_EQ(1, vec.size());
        EXPECT_EQ("abc", vec[0]);
#endif
    }
}

TEST(Test, LeadingDelimiters)
{
    {
        auto vec = std::vector<StringRef>(split(",", ","));

        ASSERT_EQ(2, vec.size());
        EXPECT_EQ("", vec[0]);
        EXPECT_EQ("", vec[1]);
    }
    {
        auto vec = std::vector<StringRef>(split(", ", ","));

        ASSERT_EQ(2, vec.size());
        EXPECT_EQ("", vec[0]);
        EXPECT_EQ(" ", vec[1]);
    }
}

TEST(Test, SimpleLiteralTests)
{
    {
        auto vec = std::vector<StringRef>(split("a", ","));

        ASSERT_EQ(1, vec.size());
        EXPECT_EQ("a", vec[0]);
    }
    {
        auto vec = std::vector<StringRef>(split("a,", ","));

        ASSERT_EQ(2, vec.size());
        EXPECT_EQ("a", vec[0]);
        EXPECT_EQ("", vec[1]);
    }
    {
        auto vec = std::vector<StringRef>(split("a,b", ","));

        ASSERT_EQ(2, vec.size());
        EXPECT_EQ("a", vec[0]);
        EXPECT_EQ("b", vec[1]);
    }
    {
        auto vec = std::vector<StringRef>(split("-a-b-c----d", "-"));

        ASSERT_EQ(8, vec.size());
        EXPECT_EQ("", vec[0]);
        EXPECT_EQ("a", vec[1]);
        EXPECT_EQ("b", vec[2]);
        EXPECT_EQ("c", vec[3]);
        EXPECT_EQ("", vec[4]);
        EXPECT_EQ("", vec[5]);
        EXPECT_EQ("", vec[6]);
        EXPECT_EQ("d", vec[7]);
    }
    {
        auto vec = std::vector<StringRef>(split("-a-b-c----d", "--"));

        ASSERT_EQ(3, vec.size());
        EXPECT_EQ("-a-b-c", vec[0]);
        EXPECT_EQ("", vec[1]);
        EXPECT_EQ("d", vec[2]);
    }
}

TEST(Test, AnyOfDelimiter)
{
    auto vec = std::vector<StringRef>(split("a.b-c,. d, e .f-", AnyOfDelimiter(".,-")));

    ASSERT_EQ(8, vec.size());
    EXPECT_EQ("a", vec[0]);
    EXPECT_EQ("b", vec[1]);
    EXPECT_EQ("c", vec[2]);
    EXPECT_EQ("", vec[3]);
    EXPECT_EQ(" d", vec[4]);
    EXPECT_EQ(" e ", vec[5]);
    EXPECT_EQ("f", vec[6]);
    EXPECT_EQ("", vec[7]);
}

TEST(Test, KeepEmpty)
{
    std::vector<StringRef> vec(split(", a ,b , c,,  ,d", ",", KeepEmpty()));

    ASSERT_EQ( 7, vec.size());
    EXPECT_EQ( "", vec[0]);
    EXPECT_EQ( " a ", vec[1]);
    EXPECT_EQ( "b ", vec[2]);
    EXPECT_EQ( " c", vec[3]);
    EXPECT_EQ( "", vec[4]);
    EXPECT_EQ( "  ", vec[5]);
    EXPECT_EQ( "d", vec[6]);
}

TEST(Test, SkipEmpty)
{
    std::vector<StringRef> vec(split(", a ,b , c,,  ,d", ",", SkipEmpty()));

    ASSERT_EQ(5, vec.size());
    EXPECT_EQ(" a ", vec[0]);
    EXPECT_EQ("b ", vec[1]);
    EXPECT_EQ(" c", vec[2]);
    EXPECT_EQ("  ", vec[3]);
    EXPECT_EQ("d", vec[4]);
}

TEST(Test, Iterators)
{
    {
        auto R = split("a,b,c,d", ",");

        std::vector<StringRef> vec;

        for (auto I = R.begin(), E = R.end(); I != E; ++I)
        {
            vec.push_back(*I);
        }

        ASSERT_EQ(4, vec.size());
        EXPECT_EQ("a", vec[0]);
        EXPECT_EQ("b", vec[1]);
        EXPECT_EQ("c", vec[2]);
        EXPECT_EQ("d", vec[3]);
    }
    {
        auto R = split("a,b,c,d", ",");

        EXPECT_EQ("a", *R.begin());
        EXPECT_EQ("a", *R.begin());
        EXPECT_EQ("a", *R.begin());

        ++R.begin();

        EXPECT_EQ("b", *R.begin());
        EXPECT_EQ("b", *R.begin());
        EXPECT_EQ("b", *R.begin());
    }
    {
        auto R = split("a,b,c,d", ",");

        auto I = R.begin();
        auto J = R.begin();

        ++I;

        EXPECT_EQ(I, I);
        EXPECT_EQ(*I, *I);
        EXPECT_EQ(J, I);
        EXPECT_EQ(*J, *I);
        EXPECT_EQ(J, J);
        EXPECT_EQ(*J, *J);

        I = ++J;

        ++J;

        EXPECT_EQ(J, I);
        EXPECT_EQ(*J, *I);
    }
}

TEST(Test, SplitOnce)
{
    {
        auto p = split_once("a:b:c:d", ":");

        EXPECT_EQ("a", p.first);
        EXPECT_EQ("b:c:d", p.second);
    }
}

TEST(Test, SplitOnceNull)
{
    {
        auto p = split_once(StringRef(), ":");

        EXPECT_TRUE(p.first.data() == nullptr);
        EXPECT_TRUE(p.second.data() == nullptr);
    }
    {
        auto p = split_once(StringRef(), ":", SkipEmpty());

        EXPECT_TRUE(p.first.data() == nullptr);
        EXPECT_TRUE(p.second.data() == nullptr);
    }
}

TEST(Test, SplitOnceEmpty)
{
    {
        auto p = split_once("", ":");

        EXPECT_TRUE(p.first.data() != nullptr);
        EXPECT_TRUE(p.second.data() == nullptr);
    }
    {
        auto p = split_once("", ":", SkipEmpty());

        EXPECT_TRUE(p.first.data() != nullptr);
        EXPECT_TRUE(p.second.data() == nullptr);
    }
}

TEST(Test, SplitOnce_A)
{
    {
        auto p = split_once("a", ":");

        EXPECT_TRUE(p.first.data() != nullptr);
        EXPECT_TRUE(p.second.data() == nullptr);
        EXPECT_EQ("a", p.first);
    }
    {
        auto p = split_once("a", ":", SkipEmpty());

        EXPECT_TRUE(p.first.data() != nullptr);
        EXPECT_TRUE(p.second.data() == nullptr);
        EXPECT_EQ("a", p.first);
    }
}

TEST(Test, SplitOnce_AS)
{
    {
        auto p = split_once("a:", ":");

        EXPECT_TRUE(p.first.data() != nullptr);
        EXPECT_TRUE(p.second.data() != nullptr);
        EXPECT_EQ("a", p.first);
        EXPECT_EQ("", p.second);
    }
    {
        auto p = split_once("a:", ":", SkipEmpty());

        EXPECT_TRUE(p.first.data() != nullptr);
        EXPECT_TRUE(p.second.data() == nullptr);
        EXPECT_EQ("a", p.first);
    }
}

TEST(Test, SplitOnce_ASS)
{
    {
        auto p = split_once("a::", ":");

        EXPECT_TRUE(p.first.data() != nullptr);
        EXPECT_TRUE(p.second.data() != nullptr);
        EXPECT_EQ("a", p.first);
        EXPECT_EQ(":", p.second);
    }
    {
        auto p = split_once("a::", ":", SkipEmpty());

        EXPECT_TRUE(p.first.data() != nullptr);
        EXPECT_TRUE(p.second.data() == nullptr);
        EXPECT_EQ("a", p.first);
    }
}

TEST(Test, SplitOnce_SA)
{
    {
        auto p = split_once(":a", ":");

        EXPECT_TRUE(p.first.data() != nullptr);
        EXPECT_TRUE(p.second.data() != nullptr);
        EXPECT_EQ("", p.first);
        EXPECT_EQ("a", p.second);
    }
    {
        auto p = split_once(":a", ":", SkipEmpty());

        EXPECT_TRUE(p.first.data() != nullptr);
        EXPECT_TRUE(p.second.data() == nullptr);
        EXPECT_EQ("a", p.first);
    }
}

TEST(Test, SplitOnce_SAS)
{
    {
        auto p = split_once(":a:", ":");

        EXPECT_TRUE(p.first.data() != nullptr);
        EXPECT_TRUE(p.second.data() != nullptr);
        EXPECT_EQ("", p.first);
        EXPECT_EQ("a:", p.second);
    }
    {
        auto p = split_once(":a:", ":", SkipEmpty());

        EXPECT_TRUE(p.first.data() != nullptr);
        EXPECT_TRUE(p.second.data() == nullptr);
        EXPECT_EQ("a", p.first);
    }
}
