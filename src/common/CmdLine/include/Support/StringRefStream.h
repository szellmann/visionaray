// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include "Support/StringRef.h"

#include <istream>
#include <streambuf>

namespace support
{

//--------------------------------------------------------------------------------------------------
// StringRefStreamBuffer
//

class StringRefStreamBuffer : public std::streambuf
{
    using BaseType = std::streambuf;

public:
    explicit StringRefStreamBuffer(StringRef str);

    // Returns the current input buffer
    StringRef strref() const { return { gptr(), egptr() }; }

protected:
    // Sets the position indicator of the inputsequence relative to some other position.
    virtual pos_type seekoff(off_type off,
                             std::ios_base::seekdir way,
                             std::ios_base::openmode which = std::ios_base::in) override;

    // Sets the position indicator of the input sequence to an absolute position.
    virtual pos_type seekpos(pos_type pos,
                             std::ios_base::openmode which = std::ios_base::in) override;

    // Reads count characters from the input sequence and stores them into a character array.
    // The characters are read as if by repeated calls to sbumpc(). That is, if less than count
    // characters are immediately available, the function calls uflow() to provide more until
    // traits::eof() is returned.
    virtual std::streamsize xsgetn(char_type* dest, std::streamsize count) override;
};

inline StringRefStreamBuffer::StringRefStreamBuffer(StringRef str)
{
    auto first = const_cast<char*>(str.data());
    setg(first, first, first + str.size());
}

inline StringRefStreamBuffer::pos_type StringRefStreamBuffer::seekoff(
        off_type off,
        std::ios_base::seekdir way,
        std::ios_base::openmode which)
{
    if (gptr() && (which & std::ios_base::in))
    {
        switch (way)
        {
        case std::ios_base::beg:
            break;
        case std::ios_base::cur:
            off += gptr() - eback();
            break;
        case std::ios_base::end:
            off += egptr() - eback();
            break;
        default:
            assert(!"not handled");
            return pos_type(-1);
        }

        if (0 <= off && off <= egptr() - eback())
        {
            setg(eback(), eback() + off, egptr());
            return pos_type(off);
        }
    }

    return pos_type(-1);
}

inline StringRefStreamBuffer::pos_type StringRefStreamBuffer::seekpos(
        pos_type pos,
        std::ios_base::openmode which)
{
    if (gptr() && (which & std::ios_base::in))
    {
        off_type Off(pos);

        if (0 <= Off && Off <= egptr() - eback())
        {
            setg(eback(), eback() + Off, egptr());
            return pos;
        }
    }

    return pos_type(-1);
}

inline std::streamsize StringRefStreamBuffer::xsgetn(char_type* dest, std::streamsize count)
{
    if (gptr() == nullptr)
        return 0;

    if (count > in_avail())
        count = in_avail();

    if (count > 0)
    {
        // Copy the characters
        traits_type::copy(dest, gptr(), static_cast<size_t>(count));

        // Advance the get pointer
        gbump(static_cast<int>(count));
    }

    return count;
}

//--------------------------------------------------------------------------------------------------
// StringRefStream
//

class StringRefStream : public std::istream
{
    using BaseType = std::istream;
    using BufferType = StringRefStreamBuffer;

    BufferType Buffer;

public:
    StringRefStream(StringRef str)
        : BaseType(nullptr)
        , Buffer(str)
    {
        init(&Buffer);
    }

    // Returns the current input buffer
    StringRef strref() const { return Buffer.strref(); }
};

} // namespace support
