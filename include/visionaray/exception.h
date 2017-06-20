// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_EXCEPTION_H
#define VSNRAY_EXCEPTION_H 1

#include <exception>
#include <string>

#include <visionaray/detail/macros.h>
#include <visionaray/export.h>

namespace visionaray
{

class VSNRAY_EXPORT exception : public std::exception
{
public:

    exception(std::string const& what = "visionaray exception");
    virtual ~exception() VSNRAY_NOEXCEPT {}

    virtual char const* what() const VSNRAY_NOEXCEPT;
    virtual char const* where() const VSNRAY_NOEXCEPT;

protected:

    std::string what_;
    std::string where_;

};


class VSNRAY_EXPORT not_implemented_yet : public exception
{
public:

    not_implemented_yet(std::string const& what = "not implemented yet")
        : exception(what)
    {
    }

};

} // visionaray

#endif // VSNRAY_EXCEPTION_H
