// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_FILE_MAPPING_H
#define VSNRAY_COMMOM_FILE_MAPPING_H 1

#include <string>

#ifdef _WIN32
#include <windows.h>
#endif

namespace visionaray {

class file_mapping
{
    void* mapping;
    size_t num_bytes;
#ifdef _WIN32
    HANDLE file;
    HANDLE mapping_handle;
#else
    int file;
#endif

public:
    file_mapping() = default;
    // Map the file into memory
    file_mapping(std::string const& fname);
    file_mapping(file_mapping&& fm);
    ~file_mapping();
    file_mapping& operator=(file_mapping&& fm);

    file_mapping(file_mapping const&) = delete;
    file_mapping& operator=(file_mapping const&) = delete;

    uint8_t const* data() const;
    size_t nbytes() const;
};

struct mapped_file
{
    mapped_file() = default;
    mapped_file(std::string const& fname);

    mapped_file& operator=(mapped_file&&) = delete;

    mapped_file(mapped_file const&) = delete;
    mapped_file& operator=(file_mapping const&) = delete;

    size_t tellg() const;
    void seek(size_t p);
    size_t read(char* buf, size_t len);

    file_mapping fm;
    std::string_view view;
    size_t pos = 0;
};

} // visionaray

#endif // VSNRAY_COMMOM_FILE_MAPPING_H
