// This file is distributed under the MIT license.
// See the LICENSE file for details

#pragma once

#ifndef VSNRAY_COMMON_ASYNC_MESSAGE_H
#define VSNRAY_COMMON_ASYNC_MESSAGE_H 1

#include <cassert>
#include <memory>
#include <stdexcept>
#include <vector>

#include <boost/uuid/uuid.hpp>

namespace visionaray
{
namespace async
{

class message;

using message_pointer = std::shared_ptr<message>;


//--------------------------------------------------------------------------------------------------
// message
//

class message
{
    friend class connection;
    friend class connection_manager;

    struct header
    {
        // The unique ID of this message
        boost::uuids::uuid id_; // POD, 16 bytes
        // The type of this message
        unsigned type_;
        // The length of this message
        unsigned size_;

        header();
        header(boost::uuids::uuid const& id, unsigned type, unsigned size);

       ~header();
    };

public:
    using data_type = std::vector<char>;

private:
    // The message data
    data_type data_;
    // The message header
    header header_;

public:
    message();

    explicit message(unsigned type);

    // Creates a message from the given buffer.
    template <typename It>
    explicit message(unsigned type, It first, It last)
        : data_(first, last)
        , header_(generate_id(), type, static_cast<unsigned>(data_.size()))
    {
    }

   ~message();

    // Returns the unique ID of this message
    boost::uuids::uuid const& id() const
    {
        return header_.id_;
    }

    // Returns the type of this message
    unsigned type() const
    {
        return header_.type_;
    }

    // Returns the size of the message
    unsigned size() const
    {
        assert( header_.size_ == data_.size() );
        return static_cast<unsigned>(data_.size());
    }

    // Returns an iterator to the first element of the data
    data_type::iterator begin()
    {
        return data_.begin();
    }

    // Returns an iterator to the element following the last element of the data
    data_type::iterator end()
    {
        return data_.end();
    }

    // Returns an iterator to the first element of the data
    data_type::const_iterator begin() const
    {
        return data_.begin();
    }

    // Returns an iterator to the element following the last element of the data
    data_type::const_iterator end() const
    {
        return data_.end();
    }

    // Swaps the data buffer with the given buffer and resets the header.
    void swap_data(data_type& buffer)
    {
        data_.swap(buffer);
        header_ = {};
    }

    // Returns a pointer to the data
    char* data()
    {
        return data_.data();
    }

    // Returns a pointer to the data
    char const* data() const
    {
        return data_.data();
    }

private:
    // Creates a new unique ID for this message
    static boost::uuids::uuid generate_id();
};

inline message_pointer make_message(unsigned type = 0)
{
    return std::make_shared<message>(type);
}

template <typename It>
inline message_pointer make_message(unsigned type, It first, It last)
{
    return std::make_shared<message>(type, first, last);
}

} // async
} // visionaray

#endif // VSNRAY_COMMON_ASYNC_MESSAGE_H
