// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cstddef>

#include "obj_grammar.h"

namespace qi = boost::spirit::qi;

using namespace visionaray;

using boost::string_ref;

namespace boost
{
namespace spirit
{
namespace traits
{

template <typename Iterator, typename Enable>
struct assign_to_attribute_from_iterators<string_ref, Iterator, Enable>
{
    static void call(Iterator const& first, Iterator const& last, string_ref& attr)
    {
        attr = { first, static_cast<size_t>(last - first) };
    }
};

} // traits
} // spirit
} // boost


obj_grammar::obj_grammar()
    : r_unhandled(*(qi::char_ - qi::eol) >> qi::eol)
    , r_text_to_eol(qi::raw[*(qi::char_ - qi::eol)])
    , r_vec3(qi::float_ >> qi::float_ >> qi::float_)
    , r_newmtl("newmtl" >> r_text_to_eol >> qi::eol)
    , r_ka("Ka" >> r_vec3 >> qi::eol)
    , r_kd("Kd" >> r_vec3 >> qi::eol)
    , r_ke("Ke" >> r_vec3 >> qi::eol)
    , r_ks("Ks" >> r_vec3 >> qi::eol)
    , r_tr("Tr" >> qi::float_ >> qi::eol)
    , r_d("d" >> qi::float_ >> qi::eol)
    , r_ns("Ns" >> qi::float_ >> qi::eol)
    , r_ni("Ni" >> qi::float_ >> qi::eol)
    , r_map_kd("map_Kd" >> r_text_to_eol >> qi::eol)
    , r_illum("illum" >> qi::int_ >> qi::eol)
    , r_comment("#" >> r_text_to_eol >> qi::eol)
    , r_g("g" >> r_text_to_eol >> qi::eol)
    , r_mtllib("mtllib" >> r_text_to_eol >> qi::eol)
    , r_usemtl("usemtl" >> r_text_to_eol >> qi::eol)
    , r_v("v" >> qi::float_ >> qi::float_ >> qi::float_ >> -qi::float_ >> qi::eol // TODO: mind w
        | "v" >> qi::float_ >> qi::float_ >> qi::float_
              >> qi::float_ >> qi::float_ >> qi::float_ >> qi::eol) // RGB color (extension)
    , r_vt("vt" >> qi::float_ >> qi::float_ >> -qi::float_ >> qi::eol) // TODO: mind w
    , r_vn("vn" >> r_vec3 >> qi::eol)
    , r_vertices(r_v >> *r_v)
    , r_tex_coords(r_vt >> *r_vt)
    , r_normals(r_vn >> *r_vn)
    , r_face_idx(qi::int_ >> -qi::lit("/") >> -qi::int_ >> -qi::lit("/") >> -qi::int_)
    , r_face("f" >> r_face_idx >> r_face_idx >> r_face_idx >> *r_face_idx >> qi::eol)
{
}
