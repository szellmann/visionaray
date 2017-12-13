// This file is distributed under the MIT license.
// See the LICENSE file for details

#include <cstdio>
#include <sstream>

#include <boost/asio/buffer.hpp>
#include <boost/asio/connect.hpp>
#include <boost/asio/placeholders.hpp>
#include <boost/asio/read.hpp>
#include <boost/asio/write.hpp>

#include <boost/thread.hpp>
#include <boost/thread/locks.hpp>

#include <boost/bind.hpp>

#include "connection_manager.h"

using namespace visionaray::async;

using boost::asio::ip::tcp;


//--------------------------------------------------------------------------------------------------
// Misc.
//

template <typename T>
inline std::string to_string(T const& x)
{
    std::ostringstream stream;
    stream << x;
    return stream.str();
}


//--------------------------------------------------------------------------------------------------
// connection_manager
//

connection_manager::connection_manager()
    : io_service_()
    , acceptor_(io_service_)
    , strand_(io_service_)
    , work_(std::make_shared<boost::asio::io_service::work>(std::ref(io_service_)))
    , connections_()
    , write_queue_()
{
}

connection_manager::connection_manager(unsigned short port)
    : io_service_()
    , acceptor_(io_service_, tcp::endpoint(tcp::v6(), port))
    , strand_(io_service_)
    , work_(std::make_shared<boost::asio::io_service::work>(std::ref(io_service_)))
    , connections_()
    , write_queue_()
{
}

connection_manager::~connection_manager()
{
    try
    {
        stop();
        close_all();
    }
    catch (std::exception const& e)
    {
        static_cast<void>(e);
    }
}

void connection_manager::run()
{
#ifndef NDEBUG
    try
    {
        io_service_.run();
    }
    catch (std::exception const& e)
    {
        printf("connection_manager::run: EXCEPTION caught: %s", e.what());
        throw;
    }
#else
    io_service_.run();
#endif
}

void connection_manager::run_in_thread()
{
    runner_ = boost::thread(&connection_manager::run, this);
}

void connection_manager::wait()
{
    runner_.join();
}

void connection_manager::stop()
{
    work_.reset();

    io_service_.stop();
    io_service_.reset();
}

void connection_manager::accept(handler h)
{
    // Start an accept operation for a new connection.
    strand_.post(boost::bind(&connection_manager::do_accept, this, h));
}

void connection_manager::connect(std::string const& host, unsigned short port, handler h)
{
    // Start a new connection operation
    strand_.post(boost::bind(&connection_manager::do_connect, this, host, port, h));
}

connection_pointer connection_manager::connect(std::string const& host, unsigned short port)
{
    using boost::asio::ip::tcp;

    connection_pointer conn(new connection(*this));

    // Resolve the host name into an IP address.
    tcp::resolver resolver(io_service_);
    tcp::resolver::query query(host, to_string(port));
    tcp::resolver::iterator I = resolver.resolve(query);
    tcp::resolver::iterator E;

    boost::system::error_code error_code;

    // Connect
    boost::asio::connect(conn->socket_, I, E, error_code);

    if (error_code)
    {
        return connection_pointer();
    }

    // Save the connection
    add_connection(conn);

    return conn;
}

connection_pointer connection_manager::get_or_connect(std::string const& host, unsigned short port)
{
    // Look for an existing connection
    connection_pointer conn = find(host, port);

    if (conn.get() == 0)
    {
        // No connection found.
        // Create a new one.
        conn = connect(host, port);
    }

    return conn;
}

void connection_manager::close(connection_pointer conn)
{
    connections::iterator I = connections_.find(conn);

    if (I == connections_.end())
    {
        return;
    }

    // Remove the handler!
    conn->remove_handler();

    // Close the connection
    conn->socket_.shutdown(tcp::socket::shutdown_both);
    conn->socket_.close();

    // Remove from the list.
    // Eventually deletes the socket.
    connections_.erase(I);
}

void connection_manager::close_all()
{
    std::for_each(connections_.begin(), connections_.end(), [&](connection_pointer conn) { close(conn); });

    connections_.clear();
}

connection_pointer connection_manager::find(std::string const& host, unsigned short port)
{
    using boost::asio::ip::tcp;

    // Get the endpoint
    tcp::resolver resolver(io_service_);
    tcp::resolver::query query(host, to_string(port));
    tcp::resolver::iterator it = resolver.resolve(query);

    tcp::endpoint endpoint = *it;

    // Check if a connection with the required endpoint exists
    for (connections::iterator I = connections_.begin(), E = connections_.end(); I != E; ++I)
    {
        tcp::endpoint remoteEndpoint = (*I)->socket_.remote_endpoint();

        if (endpoint == remoteEndpoint)
            return *I; // Found one
    }

    // There is currently no connection to the given endpoint...
    return connection_pointer();
}

//--------------------------------------------------------------------------------------------------
// Implementation
//--------------------------------------------------------------------------------------------------

void connection_manager::do_accept(handler h)
{
    connection_pointer conn(new connection(*this));

    // Start an accept operation for a new connection.
    acceptor_.async_accept(
            conn->socket_,
            boost::bind(&connection_manager::handle_accept, this, boost::asio::placeholders::error, conn, h)
            );
}

void connection_manager::handle_accept(boost::system::error_code const& e, connection_pointer conn, handler h)
{
    bool ok = h(conn, e);

    if (!e)
    {
        if (ok)
        {
            // Save the connection
            add_connection(conn);
        }
    }
    else
    {
#ifndef NDEBUG
        printf("connection_manager::handle_accept: %s", e.message().c_str());
#endif
    }
}

void connection_manager::do_connect(std::string const& host, unsigned short port, handler h)
{
    connection_pointer conn(new connection(*this));

    // Resolve the host name into an IP address.
    tcp::resolver resolver(io_service_);
    tcp::resolver::query query(host, to_string(port));
    tcp::resolver::iterator endpoint_iterator = resolver.resolve(query);

    // Start an asynchronous connect operation.
    boost::asio::async_connect(
            conn->socket_,
            endpoint_iterator,
            boost::bind(&connection_manager::handle_connect, this, boost::asio::placeholders::error, conn, h)
            );
}

void connection_manager::handle_connect(boost::system::error_code const& e, connection_pointer conn, handler h)
{
    bool ok = h(conn, e);

    if (!e)
    {
        if (ok)
        {
            // Successfully established connection.
            add_connection(conn);
        }
    }
    else
    {
#ifndef NDEBUG
        printf("connection_manager::handle_connect: %s", e.message().c_str());
#endif
    }
}

void connection_manager::do_read(connection_pointer conn)
{
    message_pointer message = make_message();

    // Issue a read operation to read exactly the number of bytes in a header.
    boost::asio::async_read(
            conn->socket_,
            boost::asio::buffer(&message->header_, sizeof(message->header_)),
            boost::bind(&connection_manager::handle_read_header, this, boost::asio::placeholders::error, message, conn)
            );
}

void connection_manager::handle_read_header(boost::system::error_code const& e, message_pointer message, connection_pointer conn)
{
    if (!e)
    {
        //
        // TODO:
        // Need to deserialize the message-header!
        //

        // Allocate memory for the message data
        message->data_.resize(message->header_.size_);

        assert( message->header_.size_ != 0 );
        assert( message->header_.size_ == message->data_.size() );

        // Start an asynchronous call to receive the data.
        boost::asio::async_read(
                conn->socket_,
                boost::asio::buffer(&message->data_[0], message->data_.size()),
                boost::bind(&connection_manager::handle_read_data, this, boost::asio::placeholders::error, message, conn)
                );
    }
    else
    {
#ifndef NDEBUG
        printf("connection_manager::handle_read_header: %s", e.message().c_str());
#endif

#if 1
        // Call the connection's slot
        conn->signal_(connection::Read, message, e);
#endif
        // Delete the connection
        remove_connection(conn);
    }
}

void connection_manager::handle_read_data(boost::system::error_code const& e, message_pointer message, connection_pointer conn)
{
    // Call the connection's slot
    conn->signal_(connection::Read, message, e);

    if (!e)
    {
        // Read the next message
        do_read(conn);
    }
    else
    {
#ifndef NDEBUG
        printf("connection_manager::handle_read_data: %s", e.message().c_str());
#endif

        remove_connection(conn);
    }
}

void connection_manager::write(message_pointer msg, connection_pointer conn)
{
    strand_.post(boost::bind(&connection_manager::do_write, this, msg, conn));
}

void connection_manager::do_write(message_pointer msg, connection_pointer conn)
{
    write_queue_.push_back(std::make_pair(conn, msg));

    if (write_queue_.size() == 1)
    {
        do_write_0();
    }
}

void connection_manager::do_write_0()
{
    // Get the next message from the queue
    std::pair<connection_pointer, message_pointer> msg = write_queue_.front();

    //
    // TODO:
    // Need to serialize the message-header!
    //

    assert( msg.second->header_.size_ != 0 );
    assert( msg.second->header_.size_ == msg.second->data_.size() );

    // Send the header and the data in a single write operation.
    std::vector<boost::asio::const_buffer> buffers;

    buffers.push_back(boost::asio::const_buffer(&msg.second->header_, sizeof(msg.second->header_)));
    buffers.push_back(boost::asio::const_buffer(&msg.second->data_[0], msg.second->data_.size()));

    // Start the write operation.
    boost::asio::async_write(
            msg.first->socket_,
            buffers,
            boost::bind(&connection_manager::handle_write, this, boost::asio::placeholders::error, msg.second, msg.first)
            );
}

void connection_manager::handle_write(boost::system::error_code const& e, message_pointer message, connection_pointer conn)
{
    // Call the connection's slot
    conn->signal_(connection::Write, message, e);

    // Remove the message from the queue
    write_queue_.pop_front();

    if (!e)
    {
        // Message successfully sent.
        // Send the next one -- if any.
        if (!write_queue_.empty())
        {
            do_write_0();
        }
    }
    else
    {
#ifndef NDEBUG
        printf("connection_manager::handle_write: %s", e.message().c_str());
#endif

        remove_connection(conn);
    }
}

void connection_manager::add_connection(connection_pointer conn)
{
    // Save the connection
    connections_.insert(conn);

    // Start reading messages
    do_read(conn);
}

void connection_manager::remove_connection(connection_pointer conn)
{
    // Delete the connection
    connections_.erase(conn);
}
