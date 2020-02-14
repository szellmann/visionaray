// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_CFILE_H
#define VSNRAY_COMMON_CFILE_H 1

#include <cassert>
#include <cstdio>
#include <string>

#include <visionaray/detail/macros.h>

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// RAII wrapper for ANSI-C FILE -- libpng, libjpeg and similar require c-style FILE handles
//

class cfile
{
public:

    cfile(std::string const& filename, std::string const& mode)
        : file_(fopen(filename.c_str(), mode.c_str()))
    {
    }

   ~cfile()
    {
        fclose(file_);
    }

    FILE* get() const { return file_; }
    bool good() const { return file_ != nullptr; }
    void close()
    {
        assert(file_);

        int res = fclose(file_);

        assert(res != EOF);

        VSNRAY_UNUSED(res);
    }

private:

    FILE* file_;

};

} // visionaray

#endif // VSNRAY_COMMON_CFILE_H
