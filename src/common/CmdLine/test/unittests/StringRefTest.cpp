// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include "Support/StringRef.h"

#include <gtest/gtest.h>

using namespace support;

template <class StringT>
void CheckFind()
{
    StringT E = "";
    StringT X = "x";
    StringT Y = "y";
    StringT S = "xxx";

    EXPECT_EQ(E.find(E),                                0);
    EXPECT_EQ(S.find(E),                                0);
    EXPECT_EQ(E.find(E, 2),                             StringT::npos);
    EXPECT_EQ(S.find(E, 2),                             2);
    EXPECT_EQ(E.find(E, 8),                             StringT::npos);
    EXPECT_EQ(S.find(E, 8),                             StringT::npos);
    EXPECT_EQ(E.find(E, StringT::npos),                 StringT::npos);
    EXPECT_EQ(S.find(E, StringT::npos),                 StringT::npos);

    EXPECT_EQ(E.find_first_of(E),                       StringT::npos);
    EXPECT_EQ(S.find_first_of(E),                       StringT::npos);
    EXPECT_EQ(E.find_first_of(E, 2),                    StringT::npos);
    EXPECT_EQ(S.find_first_of(E, 2),                    StringT::npos);
    EXPECT_EQ(E.find_first_of(E, 8),                    StringT::npos);
    EXPECT_EQ(S.find_first_of(E, 8),                    StringT::npos);
    EXPECT_EQ(E.find_first_of(E, StringT::npos),        StringT::npos);
    EXPECT_EQ(S.find_first_of(E, StringT::npos),        StringT::npos);
    EXPECT_EQ(E.find_first_of(X),                       StringT::npos);
    EXPECT_EQ(S.find_first_of(X),                       0);
    EXPECT_EQ(E.find_first_of(X, 2),                    StringT::npos);
    EXPECT_EQ(S.find_first_of(X, 2),                    2);
    EXPECT_EQ(E.find_first_of(X, 8),                    StringT::npos);
    EXPECT_EQ(S.find_first_of(X, 8),                    StringT::npos);
    EXPECT_EQ(E.find_first_of(X, StringT::npos),        StringT::npos);
    EXPECT_EQ(S.find_first_of(X, StringT::npos),        StringT::npos);
    EXPECT_EQ(E.find_first_of(Y, 2),                    StringT::npos);
    EXPECT_EQ(S.find_first_of(Y, 2),                    StringT::npos);
    EXPECT_EQ(E.find_first_of(Y, 8),                    StringT::npos);
    EXPECT_EQ(S.find_first_of(Y, 8),                    StringT::npos);
    EXPECT_EQ(E.find_first_of(Y, StringT::npos),        StringT::npos);
    EXPECT_EQ(S.find_first_of(Y, StringT::npos),        StringT::npos);

    EXPECT_EQ(E.find_first_not_of(E),                   StringT::npos);
    EXPECT_EQ(S.find_first_not_of(E),                   0);
    EXPECT_EQ(E.find_first_not_of(E, 2),                StringT::npos);
    EXPECT_EQ(S.find_first_not_of(E, 2),                2);
    EXPECT_EQ(E.find_first_not_of(E, 8),                StringT::npos);
    EXPECT_EQ(S.find_first_not_of(E, 8),                StringT::npos);
    EXPECT_EQ(E.find_first_not_of(E, StringT::npos),    StringT::npos);
    EXPECT_EQ(S.find_first_not_of(E, StringT::npos),    StringT::npos);
    EXPECT_EQ(E.find_first_not_of(X),                   StringT::npos);
    EXPECT_EQ(S.find_first_not_of(X),                   StringT::npos);
    EXPECT_EQ(E.find_first_not_of(X, 2),                StringT::npos);
    EXPECT_EQ(S.find_first_not_of(X, 2),                StringT::npos);
    EXPECT_EQ(E.find_first_not_of(X, 8),                StringT::npos);
    EXPECT_EQ(S.find_first_not_of(X, 8),                StringT::npos);
    EXPECT_EQ(E.find_first_not_of(X, StringT::npos),    StringT::npos);
    EXPECT_EQ(S.find_first_not_of(X, StringT::npos),    StringT::npos);
    EXPECT_EQ(E.find_first_not_of(Y, 2),                StringT::npos);
    EXPECT_EQ(S.find_first_not_of(Y, 2),                2);
    EXPECT_EQ(E.find_first_not_of(Y, 8),                StringT::npos);
    EXPECT_EQ(S.find_first_not_of(Y, 8),                StringT::npos);
    EXPECT_EQ(E.find_first_not_of(Y, StringT::npos),    StringT::npos);
    EXPECT_EQ(S.find_first_not_of(Y, StringT::npos),    StringT::npos);

    EXPECT_EQ(E.find_last_of(E),                        StringT::npos);
    EXPECT_EQ(S.find_last_of(E),                        StringT::npos);
    EXPECT_EQ(E.find_last_of(E, 2),                     StringT::npos);
    EXPECT_EQ(S.find_last_of(E, 2),                     StringT::npos);
    EXPECT_EQ(E.find_last_of(E, 8),                     StringT::npos);
    EXPECT_EQ(S.find_last_of(E, 8),                     StringT::npos);
    EXPECT_EQ(E.find_last_of(E, 0),                     StringT::npos);
    EXPECT_EQ(S.find_last_of(E, 0),                     StringT::npos);
    EXPECT_EQ(E.find_last_of(E),                        StringT::npos);
    EXPECT_EQ(S.find_last_of(E),                        StringT::npos);
    EXPECT_EQ(E.find_last_of(X, 2),                     StringT::npos);
    EXPECT_EQ(S.find_last_of(X, 2),                     2);
    EXPECT_EQ(E.find_last_of(X, 8),                     StringT::npos);
    EXPECT_EQ(S.find_last_of(X, 8),                     2);
    EXPECT_EQ(E.find_last_of(X, 0),                     StringT::npos);
    EXPECT_EQ(S.find_last_of(X, 0),                     0);
    EXPECT_EQ(E.find_last_of(Y, 2),                     StringT::npos);
    EXPECT_EQ(S.find_last_of(Y, 2),                     StringT::npos);
    EXPECT_EQ(E.find_last_of(Y, 8),                     StringT::npos);
    EXPECT_EQ(S.find_last_of(Y, 8),                     StringT::npos);
    EXPECT_EQ(E.find_last_of(Y, 0),                     StringT::npos);
    EXPECT_EQ(S.find_last_of(Y, 0),                     StringT::npos);

    EXPECT_EQ(E.find_last_not_of(E),                    StringT::npos);
    EXPECT_EQ(S.find_last_not_of(E),                    2);
    EXPECT_EQ(E.find_last_not_of(E, 2),                 StringT::npos);
    EXPECT_EQ(S.find_last_not_of(E, 2),                 2);
    EXPECT_EQ(E.find_last_not_of(E, 8),                 StringT::npos);
    EXPECT_EQ(S.find_last_not_of(E, 8),                 2);
    EXPECT_EQ(E.find_last_not_of(E, 0),                 StringT::npos);
    EXPECT_EQ(S.find_last_not_of(E, 0),                 0);
    EXPECT_EQ(E.find_last_not_of(E),                    StringT::npos);
    EXPECT_EQ(S.find_last_not_of(E),                    2);
    EXPECT_EQ(E.find_last_not_of(X, 2),                 StringT::npos);
    EXPECT_EQ(S.find_last_not_of(X, 2),                 StringT::npos);
    EXPECT_EQ(E.find_last_not_of(X, 8),                 StringT::npos);
    EXPECT_EQ(S.find_last_not_of(X, 8),                 StringT::npos);
    EXPECT_EQ(E.find_last_not_of(X, 0),                 StringT::npos);
    EXPECT_EQ(S.find_last_not_of(X, 0),                 StringT::npos);
    EXPECT_EQ(E.find_last_not_of(Y, 2),                 StringT::npos);
    EXPECT_EQ(S.find_last_not_of(Y, 2),                 2);
    EXPECT_EQ(E.find_last_not_of(Y, 8),                 StringT::npos);
    EXPECT_EQ(S.find_last_not_of(Y, 8),                 2);
    EXPECT_EQ(E.find_last_not_of(Y, 0),                 StringT::npos);
    EXPECT_EQ(S.find_last_not_of(Y, 0),                 0);
}

TEST(Test, CheckStdString)
{
    CheckFind<std::string>();
}

TEST(Test, CheckStringRef)
{
    CheckFind<StringRef>();
}

static void CheckSplit(StringRef str, size_t pos, size_t n, StringRef first, StringRef second)
{
    auto P = str.remove_substr(pos, n);

    auto S = P.first.str() + str.substr(pos, n) + P.second;

    ASSERT_EQ(P.first, first);
    ASSERT_EQ(P.second, second);
    ASSERT_EQ(S, str);
}

TEST(Test, Split)
{
    EXPECT_NO_FATAL_FAILURE(CheckSplit(""   ,               0,               0, ""   , ""   ));
    EXPECT_NO_FATAL_FAILURE(CheckSplit(""   , StringRef::npos,               0, ""   , ""   ));
    EXPECT_NO_FATAL_FAILURE(CheckSplit(""   ,               2, StringRef::npos, ""   , ""   ));
    EXPECT_NO_FATAL_FAILURE(CheckSplit(""   ,               1, StringRef::npos, ""   , ""   ));
    EXPECT_NO_FATAL_FAILURE(CheckSplit(""   ,               0, StringRef::npos, ""   , ""   ));
    EXPECT_NO_FATAL_FAILURE(CheckSplit(""   , StringRef::npos, StringRef::npos, ""   , ""   ));
    EXPECT_NO_FATAL_FAILURE(CheckSplit("abc",               0,               0, ""   , "abc"));
    EXPECT_NO_FATAL_FAILURE(CheckSplit("abc",               0,               1, ""   , "bc" ));
    EXPECT_NO_FATAL_FAILURE(CheckSplit("abc",               0,               2, ""   , "c"  ));
    EXPECT_NO_FATAL_FAILURE(CheckSplit("abc", StringRef::npos,               0, "abc", ""   ));
    EXPECT_NO_FATAL_FAILURE(CheckSplit("abc",               2, StringRef::npos, "ab" , ""   ));
    EXPECT_NO_FATAL_FAILURE(CheckSplit("abc",               2,               0, "ab" , "c"  ));
    EXPECT_NO_FATAL_FAILURE(CheckSplit("abc",               2,               1, "ab" , ""   ));
    EXPECT_NO_FATAL_FAILURE(CheckSplit("abc",               1, StringRef::npos, "a"  , ""   ));
    EXPECT_NO_FATAL_FAILURE(CheckSplit("abc",               1,               2, "a"  , ""   ));
    EXPECT_NO_FATAL_FAILURE(CheckSplit("abc",               1,               1, "a"  , "c"  ));
    EXPECT_NO_FATAL_FAILURE(CheckSplit("abc",               1,               0, "a"  , "bc" ));
    EXPECT_NO_FATAL_FAILURE(CheckSplit("abc",               0, StringRef::npos, ""   , ""   ));
    EXPECT_NO_FATAL_FAILURE(CheckSplit("abc", StringRef::npos, StringRef::npos, "abc", ""   ));
}

template <class StringT>
void CheckStream()
{
    {
        std::ostringstream Stream;
        Stream << StringT("hello");
        EXPECT_EQ("hello", Stream.str());
    }
    {
        std::ostringstream Stream;
        Stream << std::setw(10) << StringT("hello");
        EXPECT_EQ("     hello", Stream.str());
    }
    {
        std::ostringstream Stream;
        Stream << std::setw(3) << StringT("hello");
        EXPECT_EQ("hello", Stream.str());
    }
    {
        std::ostringstream Stream;
        Stream << std::left << std::setw(10) << StringT("hello");
        EXPECT_EQ("hello     ", Stream.str());
    }
    {
        std::ostringstream Stream;
        Stream << std::left << std::setw(3) << StringT("hello");
        EXPECT_EQ("hello", Stream.str());
    }
    {
        std::ostringstream Stream;
        Stream << std::setw(10) << std::setfill('x') << StringT("hello");
        EXPECT_EQ("xxxxxhello", Stream.str());
    }
    {
        std::ostringstream Stream;
        Stream << std::left << std::setfill('x') << std::setw(10) << StringT("hello");
        EXPECT_EQ("helloxxxxx", Stream.str());
    }
}

TEST(Test, Stream)
{
    EXPECT_NO_FATAL_FAILURE(CheckStream<std::string>());
    EXPECT_NO_FATAL_FAILURE(CheckStream<StringRef>());
}
