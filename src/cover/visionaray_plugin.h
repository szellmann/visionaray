// This file is distributed under the MIT license.
// See the LICENSE file for details.

#ifndef VSNRAY_COVER_PLUGIN_H
#define VSNRAY_COVER_PLUGIN_H

#include <memory>

#include <cover/coVRPlugin.h>

namespace visionaray { namespace cover
{

class Visionaray : public opencover::coVRPlugin, public osg::Drawable
{
public:

    Visionaray();
   ~Visionaray();

    // COVER plugin interface

    bool init();
    void preFrame();
    void expandBoundingSphere(osg::BoundingSphere &bs);

private:

    struct impl;
    std::unique_ptr<impl> impl_;

private:

    Visionaray* cloneType() const
    {
        return new Visionaray;
    }

    osg::Object* clone(const osg::CopyOp& op) const
    {
        return new Visionaray(*this, op);
    }

    Visionaray(Visionaray const& rhs, osg::CopyOp const& op = osg::CopyOp::SHALLOW_COPY);

    void drawImplementation(osg::RenderInfo& info) const;

};

}} // namespace visionaray::cover

#endif // VSNRAY_COVER_PLUGIN_H
