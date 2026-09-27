// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <string.h>
#include <fstream>
#include <stdexcept>
#ifndef _WIN32
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>
#endif
#include "file_mapping.h"

namespace visionaray
{
    file_mapping::file_mapping(std::string const& fname)
        : mapping(nullptr)
        , num_bytes(0)
    {
#ifdef _WIN32
        file = CreateFileA(fname.c_str(), GENERIC_READ,
        		FILE_SHARE_READ, nullptr, OPEN_EXISTING,
        		FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file == INVALID_HANDLE_VALUE)
        {
            throw std::runtime_error("Failed to open file " + fname);
        }

        LARGE_INTEGER file_size;
        GetFileSizeEx(file, &file_size);
        if (file_size.QuadPart == 0)
        {
            throw std::runtime_error("Cannot map 0 size file");
        }

        mapping_handle = CreateFileMapping(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (mapping_handle == INVALID_HANDLE_VALUE)
        {
            throw std::runtime_error("Failed to create file mapping for " + fname);
        }

        num_bytes = file_size.QuadPart;
        mapping = MapViewOfFile(mapping_handle, FILE_MAP_READ, 0, 0, num_bytes);
        if (!mapping)
        {
          	throw std::runtime_error("Failed to create mapped view of file " + fname);
        }
#else
        file = open(fname.c_str(), O_RDONLY);
        if (file == -1)
        {
            perror("failed opening file");
            fflush(0);
            throw std::runtime_error("Failed to open file " + fname);
        }

        struct stat stat_buf;
        fstat(file, &stat_buf);
        num_bytes = stat_buf.st_size;

        mapping = mmap(NULL, num_bytes, PROT_READ, MAP_SHARED, file, 0);
        if (!mapping)
        {
          	throw std::runtime_error("Failed to map file!");
        }
#endif
    }

    file_mapping::file_mapping(file_mapping&& fm)
        : mapping(fm.mapping)
        , num_bytes(fm.num_bytes)
        , file(fm.file)
#ifdef _WIN32
        , mapping_handle(fm.mapping_handle)
#endif
    {
        fm.mapping = nullptr;
        fm.num_bytes = 0;
#ifdef _WIN32
        fm.file = INVALID_HANDLE_VALUE;
        fm.mapping_handle = INVALID_HANDLE_VALUE;
#else
        fm.file = -1;
#endif
    }

    file_mapping::~file_mapping()
    {
        if (mapping)
        {
#ifdef _WIN32
            UnmapViewOfFile(mapping);
            CloseHandle(mapping_handle);
            CloseHandle(file);
#else
            munmap(mapping, num_bytes);
            close(file);
#endif
        }
    }

    file_mapping& file_mapping::operator=(file_mapping&& fm)
    {
        mapping = fm.mapping;
        num_bytes = fm.num_bytes;
        file = fm.file;
#ifdef _WIN32
        mapping_handle = fm.mapping_handle;
#endif

        fm.mapping = nullptr;
        fm.num_bytes = 0;
#ifdef _WIN32
        fm.file = INVALID_HANDLE_VALUE;
        fm.mapping_handle = INVALID_HANDLE_VALUE;
#else
        fm.file = -1;
#endif

        return *this;
    }

    uint8_t const* file_mapping::data() const
    {
        return static_cast<uint8_t*>(mapping);
    }

    size_t file_mapping::nbytes() const
    {
        return num_bytes;
    }

    mapped_file::mapped_file(std::string const& fname)
        : fm(fname)
        , view((char const*)fm.data(), fm.nbytes())
    {
    }

    size_t mapped_file::tellg() const
    {
        return pos;
    }

    void mapped_file::seek(size_t p)
    {
        pos = p;
    }

    size_t mapped_file::read(char* buf, size_t len)
    {
        memcpy(buf, view.data() + pos,len);
        pos += len;
        return len;
    }
} // visionaray
