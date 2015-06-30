// This file is distributed under the MIT license.
// See the LICENSE file for details.

// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n3849.html

#pragma once

#include <cassert>
#include <string>

namespace support
{

//--------------------------------------------------------------------------------------------------
// StringRef
//

class StringRef
{
public:
    using char_type = char;
    using traits_type = std::char_traits<char_type>;

    using const_reference   = char_type const&;
    using const_pointer     = char_type const*;
    using const_iterator    = char_type const*;

private:
    // The string data - an external buffer
    const_pointer Data;
    // The length of the string
    size_t Length;

    static size_t Min(size_t x, size_t y) {
        return x < y ? x : y;
    }

public:
    enum : size_t { npos = static_cast<size_t>(-1) };

public:
    // Construct an empty StringRef.
    StringRef()
        : Data(nullptr)
        , Length(0)
    {
    }

    // Construct a StringRef from a pointer and a length.
    StringRef(const_pointer Data, size_t Length)
        : Data(Data)
        , Length(Length)
    {
        assert((Data || Length == 0) && "constructing from a nullptr and a non-zero length");
    }

    // Construct a StringRef from a C-string.
    StringRef(const_pointer Str)
        : StringRef(Str, Str ? traits_type::length(Str) : 0)
    {
    }

    // Construct from two iterators
    StringRef(const_iterator Begin, const_iterator End)
        : StringRef(Begin, static_cast<size_t>(End - Begin))
    {
        assert((Begin ? Begin <= End : !End) && "invalid iterators");
    }

    // Construct a StringRef from a std::string.
    template <class A>
    StringRef(std::basic_string<char_type, traits_type, A> const& Str)
        : StringRef(Str.data(), Str.size())
    {
    }

    // Returns a pointer to the start of the string.
    // Note: The string may not be null-terminated.
    const_pointer data() const {
        return Data;
    }

    // Returns the length of the string.
    size_t size() const {
        return Length;
    }

    // Returns whether this string is null or empty.
    bool empty() const {
        return size() == 0;
    }

    // Returns an iterator to the first element of the string.
    const_iterator begin() const {
        return data();
    }

    // Returns an iterator to one element past the last element of the string.
    const_iterator end() const {
        return data() + size();
    }

    // Array access.
    const_reference operator [](size_t Index) const
    {
        assert(Index < size() && "index out of range");
        return data()[Index];
    }

    // Returns the first character of the string.
    const_reference front() const
    {
        assert(!empty() && "index out of range");
        return data()[0];
    }

    // Returns the last character of the string.
    const_reference back() const
    {
        assert(!empty() && "index out of range");
        return data()[size() - 1];
    }

    // Returns the first N characters of the string.
    StringRef front(size_t N) const
    {
        N = Min(N, size());
        return { data(), N };
    }

    // Removes the first N characters from the string.
    StringRef drop_front(size_t N) const
    {
        N = Min(N, size());
        return { data() + N, size() - N };
    }

    // Returns the last N characters of the string.
    StringRef back(size_t N) const
    {
        N = Min(N, size());
        return { data() + (size() - N), N };
    }

    // Removes the last N characters from the string.
    StringRef drop_back(size_t N) const
    {
        N = Min(N, size());
        return { data(), size() - N };
    }

    // Returns the substring [First, Last).
    StringRef slice(size_t First, size_t Last = npos) const {
        return front(Last).drop_front(First);
    }

    // Returns the sub-string [First, First + Count).
    StringRef substr(size_t First, size_t Count = npos) const {
        return drop_front(First).front(Count);
    }

    // Removes substr(Pos, N) from the current string S.
    // Returns a pair (A,B) such that S == A + substr(Pos, N) + B.
    std::pair<StringRef, StringRef> remove_substr(size_t Pos, size_t N = 0) const {
        return { front(Pos), drop_front(Pos).drop_front(N) };
    }

    // Returns whether this string is equal to another.
    bool equals(StringRef RHS) const
    {
        return size() == RHS.size()
               && 0 == traits_type::compare(data(), RHS.data(), RHS.size());
    }

    // Lexicographically compare this string with another.
    bool less(StringRef RHS) const
    {
        int c = traits_type::compare(data(), RHS.data(), Min(size(), RHS.size()));
        return c < 0 || (c == 0 && size() < RHS.size());
    }

    // Returns whether the string starts with Prefix
    bool starts_with(StringRef Prefix) const
    {
        return size() >= Prefix.size()
               && 0 == traits_type::compare(data(), Prefix.data(), Prefix.size());
    }

    // Returns whether the string ends with Suffix
    bool ends_with(StringRef Suffix) const
    {
        return size() >= Suffix.size()
               && 0 == traits_type::compare(data() + (size() - Suffix.size()), Suffix.data(), Suffix.size());
    }

    // Constructs a std::string from this StringRef.
    template <class A = std::allocator<char_type>>
    std::basic_string<char_type, traits_type, A> str() const
    {
        return std::basic_string<char_type, traits_type, A>(begin(), end());
    }

    // Explicitly convert to a std::string
    template <class A = std::allocator<char_type>>
    explicit operator std::basic_string<char_type, traits_type, A>() const
    {
        return str<A>();
    }

    // Search for the first character Ch in the sub-string [From, Length)
    size_t find(char_type Ch, size_t From = 0) const;

    // Search for the first substring Str in the sub-string [From, Length)
    size_t find(StringRef Str, size_t From = 0) const;

    // Search for the first character in the sub-string [From, Length)
    // which matches any of the characters in Chars.
    size_t find_first_of(StringRef Chars, size_t From = 0) const;

    // Search for the first character in the sub-string [From, Length)
    // which does not match any of the characters in Chars.
    size_t find_first_not_of(StringRef Chars, size_t From = 0) const;

    // Search for the last character in the sub-string [From, Length)
    // which matches any of the characters in Chars.
    size_t find_last_of(StringRef Chars, size_t From = npos) const;

    // Search for the last character in the sub-string [From, Length)
    // which does not match any of the characters in Chars.
    size_t find_last_not_of(StringRef Chars, size_t From = npos) const;
};

inline size_t StringRef::find(char_type Ch, size_t From) const
{
    if (From >= size())
        return npos;

    if (auto I = traits_type::find(data() + From, size() - From, Ch))
        return I - data();

    return npos;
}

inline size_t StringRef::find(StringRef Str, size_t From) const
{
    if (Str.size() == 1)
        return find(Str[0], From);

    if (From > size() || Str.size() > size())
        return npos;

    if (Str.empty())
        return From;

    for (auto I = From; I != size() - Str.size() + 1; ++I)
        if (traits_type::compare(data() + I, Str.data(), Str.size()) == 0)
            return I;

    return npos;
}

inline size_t StringRef::find_first_of(StringRef Chars, size_t From) const
{
    if (From >= size() || Chars.empty())
        return npos;

    for (auto I = From; I != size(); ++I)
        if (traits_type::find(Chars.data(), Chars.size(), data()[I]))
            return I;

    return npos;
}

inline size_t StringRef::find_first_not_of(StringRef Chars, size_t From) const
{
    if (From >= size())
        return npos;

    for (auto I = From; I != size(); ++I)
        if (!traits_type::find(Chars.data(), Chars.size(), data()[I]))
            return I;

    return npos;
}

inline size_t StringRef::find_last_of(StringRef Chars, size_t From) const
{
    if (Chars.empty())
        return npos;

    if (From < size())
        From++;
    else
        From = size();

    for (auto I = From; I != 0; --I)
        if (traits_type::find(Chars.data(), Chars.size(), data()[I - 1]))
            return I - 1;

    return npos;
}

inline size_t StringRef::find_last_not_of(StringRef Chars, size_t From) const
{
    if (From < size())
        From++;
    else
        From = size();

    for (auto I = From; I != 0; --I)
        if (!traits_type::find(Chars.data(), Chars.size(), data()[I - 1]))
            return I - 1;

    return npos;
}

//--------------------------------------------------------------------------------------------------
// Comparisons
//

inline bool operator ==(StringRef LHS, StringRef RHS) {
    return LHS.equals(RHS);
}

inline bool operator !=(StringRef LHS, StringRef RHS) {
    return !(LHS == RHS);
}

inline bool operator <(StringRef LHS, StringRef RHS) {
    return LHS.less(RHS);
}

inline bool operator <=(StringRef LHS, StringRef RHS) {
    return !(RHS < LHS);
}

inline bool operator >(StringRef LHS, StringRef RHS) {
    return RHS < LHS;
}

inline bool operator >=(StringRef LHS, StringRef RHS) {
    return !(LHS < RHS);
}

//--------------------------------------------------------------------------------------------------
// Formatted output
//

inline std::ostream& operator <<(std::ostream& Stream, StringRef S) {
    return Stream << S.str();
}

//--------------------------------------------------------------------------------------------------
// String operations
//

inline std::string& operator +=(std::string& LHS, StringRef RHS) {
    return LHS.append(RHS.data(), RHS.size());
}

inline std::string operator +(StringRef LHS, std::string RHS)
{
    RHS.insert(0, LHS.data(), LHS.size());
    return std::move(RHS);
}

inline std::string operator +(std::string LHS, StringRef RHS)
{
    LHS.append(RHS.data(), RHS.size());
    return std::move(LHS);
}

} // namespace support
