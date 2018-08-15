// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <algorithm>

#include <boost/filesystem.hpp>

#include "model.h"
#include "obj_loader.h"
#include "ply_loader.h"

//-------------------------------------------------------------------------------------------------
// Helpers
//

enum model_type { OBJ, PLY, Unknown };

static model_type get_type(std::string const& filename)
{
    boost::filesystem::path p(filename);


    // OBJ

    static const std::string obj_extensions[] = { ".obj", ".OBJ" };

    if (std::find(obj_extensions, obj_extensions + 2, p.extension()) != obj_extensions + 2)
    {
        return OBJ;
    }


    // PLY

    static const std::string ply_extensions[] = { ".ply", ".PLY" };

    if (std::find(ply_extensions, ply_extensions + 2, p.extension()) != ply_extensions + 2)
    {
        return PLY;
    }

    return Unknown;
}

namespace visionaray
{

bool model::load(std::string const& filename)
{
    std::string fn(filename);
    std::replace(fn.begin(), fn.end(), '\\', '/');
    model_type mt = get_type(fn);

    try
    {
        switch (mt)
        {
        case OBJ:
            load_obj(filename, *this);
            return true;

        case PLY:
            load_ply(filename, *this);
            return true;

        default:
            return false;
        }
    }
    catch (...)
    {
        return false;
    }

    return false;
}

} // visionaray
