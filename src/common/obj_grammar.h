// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_OBJ_GRAMMAR_H
#define VSNRAY_COMMON_OBJ_GRAMMAR_H 1

#include <string>

#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/fusion/include/define_struct.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/utility/string_ref.hpp>

#include <visionaray/math/forward.h>
#include <visionaray/math/vector.h>
#include <visionaray/aligned_vector.h>

//-------------------------------------------------------------------------------------------------
// boost::fusion-adapt/define some structs for parsing
//

BOOST_FUSION_DEFINE_STRUCT
(
    (visionaray), face_index_t,
    (int, vertex_index)
    (boost::optional<int>, tex_coord_index)
    (boost::optional<int>, normal_index)
)

BOOST_FUSION_ADAPT_STRUCT
(
    visionaray::vec2,
    (float, x)
    (float, y)
)

BOOST_FUSION_ADAPT_STRUCT
(
    visionaray::vec3,
    (float, x)
    (float, y)
    (float, z)
)

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Typedefs (TODO: not in namespace visionaray!)
//

using vertex_vector     = aligned_vector<vec3>;
using tex_coord_vector  = aligned_vector<vec2>;
using normal_vector     = aligned_vector<vec3>;
using face_vector       = aligned_vector<face_index_t>;


//-------------------------------------------------------------------------------------------------
// Rules
//

struct obj_grammar
{
    using It = boost::string_ref::const_iterator;
    using skip_t = boost::spirit::qi::blank_type;
    using sref_t = boost::string_ref;
    using string = std::string;
    using VV = vertex_vector;
    using TV = tex_coord_vector;
    using NV = normal_vector;
    using FV = face_vector;
    using FI = face_index_t;

    obj_grammar();


    // common rules

    boost::spirit::qi::rule<It>                   r_unhandled;
    boost::spirit::qi::rule<It, sref_t()>         r_text_to_eol;
    boost::spirit::qi::rule<It, vec3(), skip_t>   r_vec3;

    // mtl rules

    boost::spirit::qi::rule<It, sref_t(), skip_t> r_newmtl;
    boost::spirit::qi::rule<It, vec3(), skip_t>   r_ka;
    boost::spirit::qi::rule<It, vec3(), skip_t>   r_kd;
    boost::spirit::qi::rule<It, vec3(), skip_t>   r_ke;
    boost::spirit::qi::rule<It, vec3(), skip_t>   r_ks;
    boost::spirit::qi::rule<It, float(), skip_t>  r_tr;
    boost::spirit::qi::rule<It, float(), skip_t>  r_d;
    boost::spirit::qi::rule<It, float(), skip_t>  r_ns;
    boost::spirit::qi::rule<It, float(), skip_t>  r_ni;
    boost::spirit::qi::rule<It, sref_t(), skip_t> r_map_kd;
    boost::spirit::qi::rule<It, int(), skip_t>    r_illum;

    // obj rules

    boost::spirit::qi::rule<It>                   r_comment;
    boost::spirit::qi::rule<It, sref_t(), skip_t> r_g;
    boost::spirit::qi::rule<It, sref_t(), skip_t> r_mtllib;
    boost::spirit::qi::rule<It, sref_t(), skip_t> r_usemtl;

    boost::spirit::qi::rule<It, vec3(), skip_t>   r_v;
    boost::spirit::qi::rule<It, vec2(), skip_t>   r_vt;
    boost::spirit::qi::rule<It, vec3(), skip_t>   r_vn;

    boost::spirit::qi::rule<It, VV(), skip_t>     r_vertices;
    boost::spirit::qi::rule<It, TV(), skip_t>     r_tex_coords;
    boost::spirit::qi::rule<It, NV(), skip_t>     r_normals;

    boost::spirit::qi::rule<It, FI()>             r_face_idx;
    boost::spirit::qi::rule<It, FV(), skip_t>     r_face;
};

} // visionaray

#endif // VSNRAY_COMMON_OBJ_GRAMMAR_H
