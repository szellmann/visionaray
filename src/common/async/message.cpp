// This file is distributed under the MIT license.
// See the LICENSE file for details

#include <boost/version.hpp>

#if BOOST_VERSION == 106900
#define BOOST_ALLOW_DEPRECATED_HEADERS
#endif

#include <boost/uuid/uuid_generators.hpp>

#include "message.h"

using namespace visionaray::async;


//--------------------------------------------------------------------------------------------------
// message::header
//

message::header::header()
    : id_(boost::uuids::nil_uuid())
    , type_(0)
    , size_(0)
{
}

message::header::header(boost::uuids::uuid const& id, unsigned type, unsigned size)
    : id_(id)
    , type_(type)
    , size_(size)
{
}

message::header::~header()
{
}


//--------------------------------------------------------------------------------------------------
// message
//

message::message()
{
}

message::message(unsigned type)
    : data_()
    , header_(boost::uuids::nil_uuid(), type, 0)
{
}

message::~message()
{
}

boost::uuids::uuid message::generate_id()
{
    static boost::uuids::random_generator gen;
    return gen();
}
