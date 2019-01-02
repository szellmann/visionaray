// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <algorithm>

#include <boost/filesystem.hpp>

#include "moana_loader.h"
#include "model.h"
#include "obj_loader.h"
#include "ply_loader.h"
#include "vsnray_loader.h"

//-------------------------------------------------------------------------------------------------
// Helpers
//

enum model_type { Moana, OBJ, PLY, VSNRAY, Unknown };

static model_type get_type(std::string const& filename)
{
    boost::filesystem::path p(filename);


    // Moana (json files)
    // TODO: check here if this is really a "moana" file

    static const std::string moana_extensions[] = { ".json" };

    if (std::find(moana_extensions, moana_extensions + 1, p.extension()) != moana_extensions + 1)
    {
        return Moana;
    }


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

    // VSNRAY

    static const std::string vsnray_extensions[] = { ".vsnray", ".VSNRAY" };

    if (std::find(vsnray_extensions, vsnray_extensions + 2, p.extension()) != vsnray_extensions + 2)
    {
        return VSNRAY;
    }

    return Unknown;
}

namespace visionaray
{

model::model()
{
    bbox.invalidate();
}

bool model::load(std::string const& filename)
{
    std::vector<std::string> filenames(1);

    filenames[0] = filename;

    return load(filenames);
}

bool model::load(std::vector<std::string> const& filenames)
{
    if (filenames.size() < 1)
    {
        return false;
    }

    std::string fn(filenames[0]);
    std::replace(fn.begin(), fn.end(), '\\', '/');
    model_type mt = get_type(fn);

    bool same_model_type = true;

    for (size_t i = 1; i < filenames.size(); ++i)
    {
        std::string fni(filenames[i]);
        std::replace(fni.begin(), fni.end(), '\\', '/');
        model_type mti = get_type(fni);

        if (mti != mt)
        {
            same_model_type = false;
        }
    }

    try
    {
        if (same_model_type)
        {
            switch (mt)
            {
            case Moana:
                load_moana(filenames, *this);
                return true;

            case OBJ:
                load_obj(filenames, *this);
                return true;

            case PLY:
                load_ply(filenames, *this);
                return true;

            case VSNRAY:
                load_vsnray(filenames, *this);
                return true;

            default:
                return false;
            }
        }
        else
        {
            for (auto filename : filenames)
            {
                switch (mt)
                {
                case Moana:
                    load_moana(filename, *this);
                    break;

                case OBJ:
                    load_obj(filename, *this);
                    break;

                case PLY:
                    load_ply(filename, *this);
                    break;

                case VSNRAY:
                    load_vsnray(filename, *this);
                    break;

                default:
                    break;
                }
            }
        }
    }
    catch (...)
    {
        return false;
    }

    return false;
}

} // visionaray
