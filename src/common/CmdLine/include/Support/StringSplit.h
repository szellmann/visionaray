// This file is distributed under the MIT license.
// See the LICENSE file for details.

// http://isocpp.org/files/papers/n3593.html

#pragma once

#include "Support/StringRef.h"

#include <cstddef>
#include <iterator>
#include <utility>

//
// N3593
//
// A delimiter of the empty string results in each character in the input string
// becoming one element in the output collection. This is a special case. It is done to
// match the behavior of splitting using the empty string in other programming languages
// (e.g., perl).
//
#define SUPPORT_STD_SPLIT 1

namespace support
{
namespace strings
{

struct Delimiter // Return type for delimiters
{
    size_t first;
    size_t count;
};

struct Token // Return type for tokenizers
{
    size_t first;
    size_t count;
};

//--------------------------------------------------------------------------------------------------
// Delimiter
//

struct AnyOfDelimiter
{
    std::string Chars;

    explicit AnyOfDelimiter(std::string chars)
        : Chars(std::move(chars))
    {
    }

    Delimiter operator ()(StringRef str) const
    {
#if SUPPORT_STD_SPLIT
        if (Chars.empty())
        {
            if (str.size() <= 1)
                return { StringRef::npos, 0 };
            else
                return { 1, 0 };
        }
#endif

        return { str.find_first_of(Chars), 1 };
    }
};

struct LiteralDelimiter
{
    std::string Needle;

    explicit LiteralDelimiter(std::string needle)
        : Needle(std::move(needle))
    {
    }

    Delimiter operator ()(StringRef str) const
    {
        if (Needle.empty())
        {
#if SUPPORT_STD_SPLIT
            if (str.size() <= 1)
                return { StringRef::npos, 0 };
            else
                return { 1, 0 };
#else
            // Return the whole string as a token.
            // Makes LiteralDelimiter("") behave exactly as AnyOfDelimiter("").
            return { StringRef::npos, 0 };
#endif
        }

        return { str.find(Needle), Needle.size() };
    }
};

struct LineDelimiter
{
    Delimiter operator ()(StringRef Str) const
    {
        auto I = Str.find_first_of("\n\r");

        if (I == StringRef::npos)
            return { StringRef::npos, 0 };

        // Found "\n" or "\r".
        // If this "\n\r" or "\r\n" skip the other half.
        if (I + 1 < Str.size())
        {
            auto p = Str.data();

            if ((p[I + 1] == '\n' || p[I + 1] == '\r') && p[I + 1] != p[I])
                return { I, 2 };
        }

        return { I, 1 };
    }
};

struct WrapDelimiter
{
    size_t Length;

    explicit WrapDelimiter(size_t length)
        : Length(length)
    {
        assert(length != 0 && "invalid parameter");
    }

    Delimiter operator ()(StringRef Str) const
    {
        // If the string fits into the current line, just return this last line.
        if (Str.size() <= Length)
            return { StringRef::npos, 0 };

        // Otherwise, search for the first space preceding the line length.
        auto I = Str.find_last_of(" \t", Length);

        if (I != StringRef::npos)
            return { I, 1 }; // There is a space.
        else
            return { Length, 0 }; // No space in current line, break at length.
    }
};

//template <class R>
//struct Regex
//{
//    std::regex regex;
//
//    explicit Regex(std::string const& expr) : regex(expr)
//    {
//    }
//
//    R operator ()(StringRef str) const
//    {
//        auto I = std::cregex_iterator(str.begin(), str.end(), regex);
//        auto E = std::cregex_iterator();
//
//        if (I == E)
//            return { StringRef::npos, 0 };
//
//        return { I->position(), I->length() };
//    }
//};
//
//using RegexTokenizer = Regex<Token>;
//using RegexDelimiter = Regex<Delimiter>;

//--------------------------------------------------------------------------------------------------
// Predicates
//

struct KeepEmpty
{
    bool operator ()(StringRef /*tok*/) const {
        return true;
    }
};

struct SkipEmpty
{
    bool operator ()(StringRef tok) const {
        return !tok.empty();
    }
};

//--------------------------------------------------------------------------------------------------
// Implementation details
//

namespace details
{

template <class RangeT>
class Split_iterator
{
    RangeT* R;

public:
    using iterator_category     = std::input_iterator_tag;
    using value_type            = StringRef;
    using reference             = StringRef const&;
    using pointer               = StringRef const*;
    using difference_type       = ptrdiff_t;

public:
    Split_iterator(RangeT* range = nullptr)
        : R(range)
    {
    }

    reference operator *()
    {
        assert(R && "dereferencing end() iterator");
        return R->Tok;
    }

    pointer operator ->()
    {
        assert(R && "dereferencing end() iterator");
        return &R->Tok;
    }

    Split_iterator& operator ++()
    {
        assert(R && "incrementing end() iterator");

        if (R->next() == false)
            R = nullptr;

        return *this;
    }

    Split_iterator operator ++(int)
    {
        auto t = *this;
        operator ++();
        return t;
    }

    friend bool operator ==(Split_iterator lhs, Split_iterator rhs) { return lhs.R == rhs.R; }
    friend bool operator !=(Split_iterator lhs, Split_iterator rhs) { return lhs.R != rhs.R; }
};

template <class StringT, class DelimiterT, class PredicateT>
class Split_range
{
    template <class> friend class Split_iterator;

    // The string to split
    StringT Str;
    // The delimiter
    DelimiterT Delim;
    // The predicate
    PredicateT Pred;
    // The current token
    StringRef Tok;
    // The start of the rest of the string
    size_t Pos;

public:
    using iterator          = Split_iterator<Split_range>;
    using const_iterator    = Split_iterator<Split_range>;

public:
    template <class S, class D, class P>
    Split_range(S&& str, D&& delim, P&& pred)
        : Str(std::forward<S>(str))
        , Delim(std::forward<D>(delim))
        , Pred(std::forward<P>(pred))
        , Tok(Str)
        , Pos(0)
    {
        next();
    }

    iterator begin() { return iterator(this); }
    iterator end() { return iterator(); }

//#if !SUPPORT_STD_SPLIT
    template <class ContainerT>
    explicit operator ContainerT() { return ContainerT(begin(), end()); }
//#endif

    StringRef str() const
    {
        if (Pos == StringRef::npos)
            return {};

        return { Str.data() + Pos, Str.size() - Pos };
    }

private:
    //
    // N3593:
    //
    // The result of a Delimiter's find() member function must be a std::string_view referring to
    // one of the following:
    //
    // -    A substring of find()'s argument text referring to the delimiter/separator that was
    //      found.
    // -    An empty std::string_view referring to find()'s argument's end iterator, (e.g.,
    //      std::string_view(input_text.end(), 0)). This indicates that the delimiter/separator was
    //      not found.
    //
    // [Footnote: An alternative to having a Delimiter's find() function return a std::string_view
    // is to instead have it return a std::pair<size_t, size_t> where the pair's first member is the
    // position of the found delimiter, and the second member is the length of the found delimiter.
    // In this case, Not Found could be prepresented as std::make_pair(std::string_view::npos, 0).
    // ---end footnote]
    //

    bool increment(StringRef str, Delimiter sep)
    {
        if (sep.first == StringRef::npos)
        {
            // There is no further delimiter.
            // The current string is the last token.
            Tok = str;
            Pos = StringRef::npos;
        }
        else
        {
            assert(sep.first != 0 || sep.count != 0);

            // Delimiter found.
            Tok = { str.data(), sep.first };
            Pos = Pos + sep.first + sep.count;
        }

        return true;
    }

    bool increment(StringRef str, Token tok)
    {
        if (tok.first == StringRef::npos)
        {
            // No more tokens. Stop iterating.
            //..Tok = {};
            //..Pos = StringRef::npos;

            return false;
        }
        else
        {
            assert(tok.count != 0);

            // New token found.
            Tok = { str.data() + tok.first, tok.count };
            Pos = Pos + tok.first + tok.count;

            return true;
        }
    }

    bool increment()
    {
        auto S = StringRef(Str.data() + Pos, Str.size() - Pos);

        return increment(S, Delim(S));
    }

    bool next()
    {
        do
        {
            // If the current string is the last token
            // set the iterator to the past-the-end iterator.
            if (Pos == StringRef::npos)
                return false;

            // Otherwise find the next token and adjust the iterator.
            if (!increment())
                return false;
        }
        while (!Pred(Tok));

        return true;
    }
};

struct Split_string
{
    //
    // N3593:
    //
    // Rvalue support
    //
    // As described so far, std::split() may not work correctly if splitting a std::string_view that
    // refers to a temporary string. In particular, the following will not work:
    //
    //      for (std::string_view s : std::split(GetTemporaryString(), "-")) {
    //          s now refers to a temporary string that is no longer valid.
    //      }
    //
    // To address this, std::split() will move ownership of rvalues into the Range object that is
    // returned from std::split().
    //
    static auto test(std::string&&)
        -> std::string;
    static auto test(std::string const&&)
        -> std::string;
    static auto test(StringRef)
        -> StringRef;
    static auto test(char const*)
        -> StringRef;
};

template <class T>
using Split_string_t = decltype(Split_string::test(std::declval<T>()));

struct Split_delimiter
{
    //
    // N3593:
    //
    // The default delimiter when not explicitly specified is std::LiteralDelimiter
    //
    template <class T>
    static auto test(T)
        -> T;
    static auto test(std::string)
        -> LiteralDelimiter;
    static auto test(StringRef)
        -> LiteralDelimiter;
    static auto test(char const*)
        -> LiteralDelimiter;
};

template <class T>
using Split_delimiter_t = decltype(Split_delimiter::test(std::declval<T>()));

} // namespace details

//--------------------------------------------------------------------------------------------------
// split
//

//
// N3593:
//
// The std::split() algorithm takes a std::string_view and a Delimiter as arguments, and it returns
// a Range of std::string_view objects as output. The std::string_view objects in the returned Range
// will refer to substrings of the input text. The Delimiter object defines the boundaries between
// the returned substrings.
//

template <class S, class D, class P = KeepEmpty>
auto split(S&& str, D delim, P pred = P())
    -> details::Split_range< details::Split_string_t<S>, details::Split_delimiter_t<D>, P >
{
    using R = details::Split_range< details::Split_string_t<S>, details::Split_delimiter_t<D>, P >;

    return R(std::forward<S>(str), std::forward<D>(delim), std::move(pred));
}

template <class S, class D, class P = KeepEmpty>
auto split_once(S&& str, D delim, P pred = P())
    -> std::pair<StringRef, StringRef>
{
    auto range = split(std::forward<S>(str), std::move(delim), std::move(pred));

    auto I = range.begin();

    // Copy the first token
    auto first = *I;
    // Copy the remaining string
    auto last = range.str();

    return { first, (++I == range.end()) ? StringRef() : last };
}

} // namespace strings
} // namespace support
