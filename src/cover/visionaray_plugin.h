// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COVER_PLUGIN_H
#define VSNRAY_COVER_PLUGIN_H

#include <memory>

#include <cover/coVRPlugin.h>

namespace visionaray { namespace cover {

class Visionaray : public opencover::coVRPlugin
{
public:

    Visionaray();
   ~Visionaray();

    // COVER plugin interface

    bool init();
    void preFrame();
    void expandBoundingSphere(osg::BoundingSphere &bs);
    void key(int type, int key_sym, int /* mod */);

private:

    struct impl;
    std::unique_ptr<impl> impl_;

};

}} // namespace visionaray::cover

#endif // VSNRAY_COVER_PLUGIN_H
