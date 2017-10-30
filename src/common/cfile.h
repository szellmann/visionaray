// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_CFILE_H
#define VSNRAY_COMMON_CFILE_H 1

#include <cstdio>
#include <string>

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
    bool good() const { return file_ != 0; }

private:

    FILE* file_;

};

} // visionaray

#endif // VSNRAY_COMMON_CFILE_H
