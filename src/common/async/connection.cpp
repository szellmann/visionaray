// This file is distributed under the MIT license.
// See the LICENSE file for details

#ifndef NDEBUG
#include <iostream>
#include <ostream>
#endif

#include <boost/asio/buffer.hpp>
#include <boost/asio/connect.hpp>
#include <boost/asio/placeholders.hpp>
#include <boost/asio/read.hpp>
#include <boost/asio/write.hpp>

#include "connection.h"
#include "connection_manager.h"

using namespace visionaray::async;

using boost::asio::ip::tcp;


//--------------------------------------------------------------------------------------------------
// connection
//

connection::connection(connection_manager& manager)
    : manager_(manager)
    , socket_(manager.io_service_)
{
#ifndef NDEBUG
    std::cout << "connection::connection [" << (void*)this << "]\n";
#endif
}

connection::~connection()
{
#ifndef NDEBUG
    std::cout << "connection::~connection [" << (void*)this << "]\n";
#endif

#if 0
    close(); // NO!!!
#else
    remove_handler();
#endif
}

void connection::start()
{
}

void connection::stop()
{
}

void connection::set_handler(signal_type::slot_function_type handler)
{
    // Remove existing handler.
    // Only a single handler is currently supported.
    remove_handler();

    slot_ = signal_.connect(handler);
}

void connection::remove_handler()
{
    signal_.disconnect(slot_);
}

void connection::close()
{
    manager_.close(shared_from_this());
}

void connection::write(message_pointer message)
{
    manager_.write(message, shared_from_this());
}
