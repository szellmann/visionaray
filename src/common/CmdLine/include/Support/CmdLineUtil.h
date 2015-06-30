// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <cctype>
#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace support
{
namespace cl
{

//--------------------------------------------------------------------------------------------------
// Parses a single argument from a command line string.
// Using Unix-style escaping.
//
template <class InputIterator, class OutputIterator>
std::pair<InputIterator, OutputIterator>
tokenizeStepUnix(InputIterator first, InputIterator last, OutputIterator out)
{
    //
    // See:
    //
    // http://www.gnu.org/software/bash/manual/bashref.html#Quoting
    // http://wiki.bash-hackers.org/syntax/quoting
    //

    decltype(*first) quoteChar = 0;

    for (; first != last; ++first)
    {
        auto ch = *first;

        if (quoteChar == '\\') // Quoting a single character using the backslash?
        {
            *out++ = ch;
            quoteChar = 0;
        }
        else if (quoteChar && ch != quoteChar) // Currently quoting using ' or "?
        {
            *out++ = ch;
        }
        else if (ch == '\'' || ch == '\"' || ch == '\\') // Toggle quoting?
        {
            quoteChar = quoteChar ? 0 : ch;
        }
        else if (std::isspace(ch)) // Arguments are separated by whitespace
        {
            return { ++first, out };
        }
        else // Nothing special...
        {
            *out++ = ch;
        }
    }

    return { first, out };
}

//--------------------------------------------------------------------------------------------------
// Parses a command line string and returns a list of command line arguments.
// Using Unix-style escaping.
//
template <
    class InputIterator,
    class OutputIterator,
    class String = std::basic_string<typename std::iterator_traits<InputIterator>::value_type>
    >
OutputIterator
tokenizeCommandLineUnix(InputIterator first, InputIterator last, OutputIterator out, String arg = String())
{
    while (first != last)
    {
        arg.clear();

        first = tokenizeStepUnix(first, last, std::back_inserter(arg)).first;

        if (!arg.empty())
        {
            *out++ = arg;
        }
    }

    return out;
}

struct TokenizeUnix
{
    template <
        class InputIterator,
        class OutputIterator,
        class String = std::basic_string<typename std::iterator_traits<InputIterator>::value_type>
        >
    OutputIterator
    operator ()(InputIterator first, InputIterator last, OutputIterator out, String arg = String()) const
    {
        return tokenizeCommandLineUnix(first, last, out, arg);
    }
};

//--------------------------------------------------------------------------------------------------
// Parses a single argument from a command line string.
// Using Windows-style escaping.
//
template <class InputIterator, class OutputIterator>
std::pair<InputIterator, OutputIterator>
tokenizeStepWindows(InputIterator first, InputIterator last, OutputIterator out, bool& quoting)
{
    unsigned numBackslashes = 0;

    bool recentlyClosed = false;

    quoting = false;

    for (; first != last; ++first)
    {
        auto ch = *first;

        if (ch == '\"' && recentlyClosed)
        {
            recentlyClosed = false;

            //
            // If a closing " is followed immediately by another ", the 2nd " is accepted literally
            // and added to the parameter.
            //
            // See:
            // http://www.daviddeley.com/autohotkey/parameters/parameters.htm#WINCRULESDOC
            //
            *out++ = '\"';
        }
        else if (ch == '\"')
        {
            //
            // If an even number of backslashes is followed by a double quotation mark, one backslash
            // is placed in the argv array for every pair of backslashes, and the double quotation mark
            // is interpreted as a string delimiter.
            //
            // If an odd number of backslashes is followed by a double quotation mark, one backslash
            // is placed in the argv array for every pair of backslashes, and the double quotation mark
            // is "escaped" by the remaining backslash, causing a literal double quotation mark (")
            // to be placed in argv.
            //

            bool even = (numBackslashes % 2) == 0;

            for (numBackslashes /= 2; numBackslashes != 0; --numBackslashes)
            {
                *out++ = '\\';
            }

            if (even)
            {
                recentlyClosed = quoting; // Remember if this is a closing "
                quoting = !quoting;
            }
            else
            {
                *out++ = '\"';
            }
        }
        else if (ch == '\\')
        {
            recentlyClosed = false;

            ++numBackslashes;
        }
        else
        {
            recentlyClosed = false;

            //
            // Backslashes are interpreted literally, unless they immediately precede a double
            // quotation mark.
            //
            for (; numBackslashes != 0; --numBackslashes)
            {
                *out++ = '\\';
            }

            if (!quoting && std::isspace(ch))
            {
                //
                // Arguments are delimited by white space, which is either a space or a tab.
                //
                // A string surrounded by double quotation marks ("string") is interpreted as a
                // single argument, regardless of white space contained within. A quoted string can
                // be embedded in an argument.
                //
                return { ++first, out };
            }

            *out++ = ch;
        }
    }

    return { first, out };
}

//--------------------------------------------------------------------------------------------------
// Parses a command line string and returns a list of command line arguments.
// Using Windows-style escaping.
//
template <
    class InputIterator,
    class OutputIterator,
    class String = std::basic_string<typename std::iterator_traits<InputIterator>::value_type>
    >
OutputIterator
tokenizeCommandLineWindows(InputIterator first, InputIterator last, OutputIterator out, String arg = String())
{
    while (first != last)
    {
        bool quoting = false;

        arg.clear();

        first = tokenizeStepWindows(first, last, std::back_inserter(arg), quoting).first;

        if (!arg.empty() || (/*first == last &&*/ quoting))
        {
            *out++ = arg;
        }
    }

    return out;
}

struct TokenizeWindows
{
    template <
        class InputIterator,
        class OutputIterator,
        class String = std::basic_string<typename std::iterator_traits<InputIterator>::value_type>
        >
    OutputIterator operator ()(InputIterator first, InputIterator last, OutputIterator out, String arg = String()) const
    {
        return tokenizeCommandLineWindows(first, last, out, arg);
    }
};

//--------------------------------------------------------------------------------------------------
// Quote a single command line argument. Using Windows-style escaping.
//
// See:
// http://blogs.msdn.com/b/twistylittlepassagesallalike/archive/2011/04/23/everyone-quotes-arguments-the-wrong-way.aspx
//
// This routine appends the given argument to a command line such that CommandLineToArgvW will
// return the argument string unchanged. Arguments in a command line should be separated by spaces;
// this function does not add these spaces.
//
template <class InputIterator, class OutputIterator>
OutputIterator
quoteSingleArgWindows(InputIterator first, InputIterator last, OutputIterator out)
{
    unsigned numBackslashes = 0;

    *out++ = '\"';

    for (; first != last; ++first)
    {
        auto ch = *first;

        if (ch == '\\')
        {
            ++numBackslashes;
        }
        else if (ch == '\"')
        {
            //
            // Escape all backslashes and the following
            // double quotation mark.
            //
            for (++numBackslashes; numBackslashes != 0; --numBackslashes)
            {
                *out++ = '\\';
            }
        }
        else
        {
            //
            // Backslashes aren't special here.
            //
            numBackslashes = 0;
        }

        *out++ = ch;
    }

    //
    // Escape all backslashes, but let the terminating
    // double quotation mark we add below be interpreted
    // as a metacharacter.
    //
    for (; numBackslashes != 0; --numBackslashes)
    {
        *out++ = '\\';
    }

    *out++ = '\"';

    return out;
}

//--------------------------------------------------------------------------------------------------
// Quote command line arguments. Using Windows-style escaping.
//
template <class InputIterator, class OutputIterator>
std::pair<InputIterator, OutputIterator>
quoteArgsWindows(InputIterator first, InputIterator last, OutputIterator out)
{
    using std::begin;
    using std::end;

    for (; first != last; ++first)
    {
        auto I = begin(*first);
        auto E = end(*first);

        if (I == E)
        {
            //
            // If a command line string ends while currently quoting, CommandLineToArgvW
            // will add the current argument to the argv array regardless whether the current
            // argument is empty or not.
            //
            // So if we have an empty argument here add an opening " and return. This should
            // be the last argument though...
            //
            // XXX:
            // return { last, out }; ?!?!?!
            //
            *out++ = '\"';
            break;
        }

        //
        // Append the current argument
        //
        out = quoteSingleArgWindows(I, E, out);

        //
        // Separate arguments with spaces
        //
        *out++ = ' ';
    }

    return { first, out };
}

//--------------------------------------------------------------------------------------------------
// Expand a single response file at the given position.
//
template <class Container, class Tokenizer>
typename Container::iterator
expandResponseFile(Container& cont, typename Container::iterator at, Tokenizer tokenize)
{
    using std::begin;
    using std::end;

    using Buffer = typename Container::value_type;

    auto off = std::distance(begin(cont), at);
    auto len = cont.size() - 1;

    std::ifstream file;

    file.exceptions(std::ios::failbit);
    file.open(at->data() + 1);

    // Erase the i-th argument (@file)
    cont.erase(at);

    // Parse the file while inserting new command line arguments at the end
    tokenize(std::istreambuf_iterator<char>(file.rdbuf()),
             std::istreambuf_iterator<char>(),
             std::back_inserter(cont),
             Buffer()
             );

    auto I = std::next(begin(cont), off);

    // Move the new arguments to the correct position
    std::rotate(I, std::next(begin(cont), len), end(cont));

    return I;
}

//--------------------------------------------------------------------------------------------------
// Recursively expand response files.
//
template <class Container, class Tokenizer>
void expandResponseFiles(Container& args, Tokenizer tokenize, size_t maxResponseFiles = 100)
{
    using std::begin;
    using std::end;

    // Recursively expand respond files...
    for (auto I = begin(args); I != end(args); )
    {
        if (I->empty() || I->front() != '@')
        {
            ++I;
            continue;
        }

        if (maxResponseFiles == 0)
            throw std::runtime_error("too many response files encountered");

        I = expandResponseFile(args, I, tokenize);

        maxResponseFiles--;
    }
}

//--------------------------------------------------------------------------------------------------
// Expand wild cards.
// This is only supported for Windows.
//
void expandWildcards(std::vector<std::string>& args);

} // namespace cl
} // namespace support
