// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include "Support/CmdLine.h"
#include "Support/StringSplit.h"

#include "PrettyPrint.h"

#include <functional>
#include <iostream>
#include <set>
#include <sstream>

#include <gtest/gtest.h>

using namespace support;

typedef std::vector<std::string> Argv;

bool parse(cl::CmdLine& cmd, Argv const& argv)
{
    try
    {
        cmd.parse(argv);
        return true;
    }
    catch (std::exception& e)
    {
        std::cout << "ERROR: " << e.what() << std::endl;
        return false;
    }
}

template <class T>
static std::string to_pretty_string(T const& object)
{
    std::ostringstream str;
    str << pretty(object);
    return str.str();
}

TEST(CmdLineTest, Flags1)
{
    using Pair = std::pair<unsigned, int>;

    auto test = [](bool result, Argv const& argv, Pair const& a_val, Pair const& b_val, Pair const& c_val)
    {
        SCOPED_TRACE("parsing: " + to_pretty_string(argv));

        cl::CmdLine cmd;

        auto a = cl::makeOption<bool>(cl::Parser<>(), cmd, "a");
        auto b = cl::makeOption<bool>(cl::Parser<>(), cmd, "b", cl::Grouping);
        auto c = cl::makeOption<bool>(cl::Parser<>(), cmd, "c", cl::Grouping, cl::ZeroOrMore);

        bool actual_result = parse(cmd, argv);
        EXPECT_EQ(result, actual_result);

        EXPECT_EQ(a_val.first, a->count());
        EXPECT_EQ(b_val.first, b->count());
        EXPECT_EQ(c_val.first, c->count());

        if (a->count())
            EXPECT_EQ(a_val.second, +a->value());
        if (b->count())
            EXPECT_EQ(b_val.second, +b->value());
        if (c->count())
            EXPECT_EQ(c_val.second, +c->value());
    };

    EXPECT_NO_FATAL_FAILURE( test(true,  { "-a"                   }, {1,1}, {0,0}, {0,0} ) );
    EXPECT_NO_FATAL_FAILURE( test(true,  { "-a=1"                 }, {1,1}, {0,0}, {0,0} ) );
    EXPECT_NO_FATAL_FAILURE( test(true,  { "-a=true"              }, {1,1}, {0,0}, {0,0} ) );
    EXPECT_NO_FATAL_FAILURE( test(true,  { "-a=0"                 }, {1,0}, {0,0}, {0,0} ) );
    EXPECT_NO_FATAL_FAILURE( test(true,  { "-a=false"             }, {1,0}, {0,0}, {0,0} ) );
    EXPECT_NO_FATAL_FAILURE( test(false, { "-a0"                  }, {0,0}, {0,0}, {0,0} ) ); // unknown option -a0
    EXPECT_NO_FATAL_FAILURE( test(false, { "-a1"                  }, {0,0}, {0,0}, {0,0} ) ); // unknown option -a1
    EXPECT_NO_FATAL_FAILURE( test(false, { "-ax"                  }, {0,0}, {0,0}, {0,0} ) ); // unknown option -ax
    EXPECT_NO_FATAL_FAILURE( test(true,  { "-a", "-b"             }, {1,1}, {1,1}, {0,0} ) );
    EXPECT_NO_FATAL_FAILURE( test(true,  { "-a", "-b", "-c"       }, {1,1}, {1,1}, {1,1} ) );
    EXPECT_NO_FATAL_FAILURE( test(true,  { "-a", "-bc"            }, {1,1}, {1,1}, {1,1} ) );
    EXPECT_NO_FATAL_FAILURE( test(false, { "-a", "--bc"           }, {1,1}, {0,0}, {0,0} ) ); // unknown option --bc
    EXPECT_NO_FATAL_FAILURE( test(true,  { "-a", "-cb"            }, {1,1}, {1,1}, {1,1} ) );
    EXPECT_NO_FATAL_FAILURE( test(false, { "-a", "-bcb"           }, {1,1}, {1,1}, {1,1} ) ); // -b only allowed once
    EXPECT_NO_FATAL_FAILURE( test(true,  { "-a", "-bcc"           }, {1,1}, {1,1}, {2,1} ) ); // -b only allowed once
}

TEST(CmdLineTest, Grouping1)
{
    using Pair = std::pair<unsigned, int>;

    auto test = [](bool result, Argv const& argv, Pair const& a_val, Pair const& b_val, Pair const& c_val)
    {
        SCOPED_TRACE("parsing: " + to_pretty_string(argv));

        cl::CmdLine cmd;

        auto a = cl::makeOption<bool>(cl::Parser<>(), cmd, "a", cl::Grouping, cl::ZeroOrMore);
        auto b = cl::makeOption<bool>(cl::Parser<>(), cmd, "b", cl::Grouping);
        auto c = cl::makeOption<bool>(cl::Parser<>(), cmd, "ab", cl::Prefix);

        bool actual_result = parse(cmd, argv);
        EXPECT_EQ(result, actual_result);

        EXPECT_EQ(a_val.first, a->count());
        EXPECT_EQ(b_val.first, b->count());
        EXPECT_EQ(c_val.first, c->count());

        if (a->count())
            EXPECT_EQ(a_val.second, +a->value());
        if (b->count())
            EXPECT_EQ(b_val.second, +b->value());
        if (c->count())
            EXPECT_EQ(c_val.second, +c->value());
    };

    EXPECT_NO_FATAL_FAILURE( test(true,  { "-a"                   }, {1,1}, {0,0}, {0,0} ) );
    EXPECT_NO_FATAL_FAILURE( test(true,  { "-a=1"                 }, {1,1}, {0,0}, {0,0} ) );
    EXPECT_NO_FATAL_FAILURE( test(true,  { "-a=true"              }, {1,1}, {0,0}, {0,0} ) );
    EXPECT_NO_FATAL_FAILURE( test(true,  { "-a=0"                 }, {1,0}, {0,0}, {0,0} ) );
    EXPECT_NO_FATAL_FAILURE( test(true,  { "-a=false"             }, {1,0}, {0,0}, {0,0} ) );
    EXPECT_NO_FATAL_FAILURE( test(false, { "-a0"                  }, {0,0}, {0,0}, {0,0} ) ); // unknown option -a0
    EXPECT_NO_FATAL_FAILURE( test(false, { "-a1"                  }, {0,0}, {0,0}, {0,0} ) ); // unknown option -a1
    EXPECT_NO_FATAL_FAILURE( test(false, { "-ax"                  }, {0,0}, {0,0}, {0,0} ) ); // unknown option -ax
    EXPECT_NO_FATAL_FAILURE( test(true,  { "-ab"                  }, {0,0}, {0,0}, {1,1} ) );
    EXPECT_NO_FATAL_FAILURE( test(false, { "-abb"                 }, {0,0}, {0,0}, {0,0} ) ); // invalid value for -ab
    EXPECT_NO_FATAL_FAILURE( test(true,  { "-abtrue"              }, {0,0}, {0,0}, {1,1} ) );
    EXPECT_NO_FATAL_FAILURE( test(true,  { "-abfalse"             }, {0,0}, {0,0}, {1,0} ) );
    EXPECT_NO_FATAL_FAILURE( test(true,  { "-ba"                  }, {1,1}, {1,1}, {0,0} ) );
    EXPECT_NO_FATAL_FAILURE( test(false, { "--ba"                 }, {0,0}, {0,0}, {0,0} ) ); // no check for option group
    EXPECT_NO_FATAL_FAILURE( test(true,  { "-baa"                 }, {2,1}, {1,1}, {0,0} ) ); // check for option group
    EXPECT_NO_FATAL_FAILURE( test(false, { "--baa"                }, {0,0}, {0,0}, {0,0} ) ); // no check for option group
    EXPECT_NO_FATAL_FAILURE( test(true,  { "-ba", "-a"            }, {2,1}, {1,1}, {0,0} ) );
    EXPECT_NO_FATAL_FAILURE( test(false, { "--ba", "-a"           }, {0,0}, {0,0}, {0,0} ) ); // no check for option group
    EXPECT_NO_FATAL_FAILURE( test(true,  { "-ab", "-ba"           }, {1,1}, {1,1}, {1,1} ) );
    EXPECT_NO_FATAL_FAILURE( test(true,  { "-ab1", "-ba"          }, {1,1}, {1,1}, {1,1} ) );
    EXPECT_NO_FATAL_FAILURE( test(false, { "-ab=1", "-ba"         }, {0,0}, {0,0}, {0,0} ) ); // invalid value for -ab
    EXPECT_NO_FATAL_FAILURE( test(false, { "-ab", "1", "-ba"      }, {0,0}, {0,0}, {1,1} ) ); // unhandled positional
}

TEST(CmdLineTest, Grouping2)
{
    using PairI = std::pair<unsigned, int>;
    using PairS = std::pair<unsigned, std::string>;

    auto test = [](bool result, Argv const& argv, PairI const& a_val, PairI const& b_val, PairS const& c_val)
    {
        SCOPED_TRACE("parsing: " + to_pretty_string(argv));

        cl::CmdLine cmd;

        auto a = cl::makeOption<bool>(cl::Parser<>(), cmd, "x", cl::Grouping, cl::ArgDisallowed, cl::ZeroOrMore);
        auto b = cl::makeOption<bool>(cl::Parser<>(), cmd, "v", cl::Grouping, cl::ArgDisallowed);
        auto c = cl::makeOption<std::string>(cl::Parser<>(), cmd, "f", cl::Grouping, cl::ArgRequired);

        bool actual_result = parse(cmd, argv);
        EXPECT_EQ(result, actual_result);

        EXPECT_EQ(a_val.first, a->count());
        EXPECT_EQ(b_val.first, b->count());
        EXPECT_EQ(c_val.first, c->count());

        if (a->count())
            EXPECT_EQ(a_val.second, +a->value());
        if (b->count())
            EXPECT_EQ(b_val.second, +b->value());
        if (c->count())
            EXPECT_EQ(c_val.second, c->value());
    };

    EXPECT_NO_FATAL_FAILURE( test(true,  { "-xvf", "test.tar"       }, {1,1}, {1,1}, {1,"test.tar"} ) );
    EXPECT_NO_FATAL_FAILURE( test(true,  { "-xv", "-f", "test.tar"  }, {1,1}, {1,1}, {1,"test.tar"} ) );
    EXPECT_NO_FATAL_FAILURE( test(true,  { "-xv", "-f=test.tar"     }, {1,1}, {1,1}, {1,"test.tar"} ) );
    EXPECT_NO_FATAL_FAILURE( test(false, { "-xfv", "test.tar"       }, {0,0}, {0,0}, {0,""        } ) );
}

TEST(CmdLineTest, Prefix)
{
    using Pair = std::pair<unsigned, std::string>;

    auto test = [](bool result, Argv const& argv, Pair const& r_val, Pair const& o_val)
    {
        SCOPED_TRACE("parsing: " + to_pretty_string(argv));

        cl::CmdLine cmd;

        auto r = cl::makeOption<std::string>(cl::Parser<>(), cmd, "r", cl::Prefix, cl::ArgRequired);
        auto o = cl::makeOption<std::string>(cl::Parser<>(), cmd, "o", cl::Prefix, cl::ArgOptional);

        bool actual_result = parse(cmd, argv);
        EXPECT_EQ(result, actual_result);

        EXPECT_EQ(r_val.first, r->count());
        EXPECT_EQ(o_val.first, o->count());

        if (r->count())
            EXPECT_EQ(r_val.second, r->value());
        if (o->count())
            EXPECT_EQ(o_val.second, o->value());
    };

    EXPECT_NO_FATAL_FAILURE( test(true,  {              }, {0,""        }, {0,""        }) );
    EXPECT_NO_FATAL_FAILURE( test(false, {"-r"          }, {0,""        }, {0,""        }) ); // missing argument for r
    EXPECT_NO_FATAL_FAILURE( test(false, {"-r", "x"     }, {0,""        }, {0,""        }) ); // unhandled positional arg
    EXPECT_NO_FATAL_FAILURE( test(true,  {"-rx"         }, {1,"x"       }, {0,""        }) );
    EXPECT_NO_FATAL_FAILURE( test(true,  {"-r=x"        }, {1,"=x"      }, {0,""        }) );
    EXPECT_NO_FATAL_FAILURE( test(true,  {"-r-o"        }, {1,"-o"      }, {0,""        }) );
//  EXPECT_NO_FATAL_FAILURE( test(false, {"-r", "-o"    }, {0,""        }, {1,""        }) ); // -o is a valid option
//  EXPECT_NO_FATAL_FAILURE( test(false, {"-r", "-ox"   }, {0,""        }, {1,"x"       }) ); // -o is a valid option
    EXPECT_NO_FATAL_FAILURE( test(true,  {"-o"          }, {0,""        }, {1,""        }) );
    EXPECT_NO_FATAL_FAILURE( test(false, {"-o", "x"     }, {0,""        }, {1,""        }) ); // unhandled positional arg
    EXPECT_NO_FATAL_FAILURE( test(true,  {"-ox"         }, {0,""        }, {1,"x"       }) );
    EXPECT_NO_FATAL_FAILURE( test(true,  {"-o=x"        }, {0,""        }, {1,"=x"      }) );
    EXPECT_NO_FATAL_FAILURE( test(true,  {"-o-r"        }, {0,""        }, {1,"-r"      }) );
}

TEST(CmdLineTest, MayPrefix)
{
    using Pair = std::pair<unsigned, std::string>;

    auto test = [](bool result, Argv const& argv, Pair const& r_val, Pair const& o_val)
    {
        SCOPED_TRACE("parsing: " + to_pretty_string(argv));

        cl::CmdLine cmd;

        auto r = cl::makeOption<std::string>(cl::Parser<>(), cmd, "r", cl::MayPrefix, cl::ArgRequired);
        auto o = cl::makeOption<std::string>(cl::Parser<>(), cmd, "o", cl::MayPrefix, cl::ArgOptional);

        bool actual_result = parse(cmd, argv);
        EXPECT_EQ(result, actual_result);

        EXPECT_EQ(r_val.first, r->count());
        EXPECT_EQ(o_val.first, o->count());

        if (r->count())
            EXPECT_EQ(r_val.second, r->value());
        if (o->count())
            EXPECT_EQ(o_val.second, o->value());
    };

    EXPECT_NO_FATAL_FAILURE( test(true,  {              }, {0,""        }, {0,""        }) );
    EXPECT_NO_FATAL_FAILURE( test(false, {"-r"          }, {0,""        }, {0,""        }) ); // missing argument for r
    EXPECT_NO_FATAL_FAILURE( test(true,  {"-r", "x"     }, {1,"x"       }, {0,""        }) );
    EXPECT_NO_FATAL_FAILURE( test(true,  {"-rx"         }, {1,"x"       }, {0,""        }) );
    EXPECT_NO_FATAL_FAILURE( test(true,  {"-r=x"        }, {1,"=x"      }, {0,""        }) );
    EXPECT_NO_FATAL_FAILURE( test(true,  {"-r-o"        }, {1,"-o"      }, {0,""        }) );
//  EXPECT_NO_FATAL_FAILURE( test(false, {"-r", "-o"    }, {0,""        }, {1,""        }) ); // -o is a valid option
    EXPECT_NO_FATAL_FAILURE( test(true,  {"-r", "-ox"   }, {1,"-ox"     }, {0,""        }) ); // -ox is *NOT* a valid option (quick test)
    EXPECT_NO_FATAL_FAILURE( test(true,  {"-o"          }, {0,""        }, {1,""        }) );
    EXPECT_NO_FATAL_FAILURE( test(false, {"-o", "x"     }, {0,""        }, {1,""        }) ); // unhandled positional arg
    EXPECT_NO_FATAL_FAILURE( test(true,  {"-ox"         }, {0,""        }, {1,"x"       }) );
    EXPECT_NO_FATAL_FAILURE( test(true,  {"-o=x"        }, {0,""        }, {1,"=x"      }) );
    EXPECT_NO_FATAL_FAILURE( test(true,  {"-o-r"        }, {0,""        }, {1,"-r"      }) );
}

TEST(CmdLineTest, Equals)
{
    auto test = [](Argv const& argv, std::string const& val) -> bool
    {
        SCOPED_TRACE("parsing: " + to_pretty_string(argv));

        cl::CmdLine cmd;

        auto a = cl::makeOption<std::string>(cl::Parser<>(), cmd, "a", cl::Prefix, cl::ArgRequired);
        auto b = cl::makeOption<std::string>(cl::Parser<>(), cmd, "b", cl::Prefix, cl::ArgOptional);
        auto c = cl::makeOption<std::string>(cl::Parser<>(), cmd, "c", cl::ArgRequired);
        auto d = cl::makeOption<std::string>(cl::Parser<>(), cmd, "d", cl::ArgOptional);

        if (!parse(cmd, argv))
            return false;

        if (a->count()) EXPECT_EQ(a->value(), val);
        if (b->count()) EXPECT_EQ(b->value(), val);
        if (c->count()) EXPECT_EQ(c->value(), val);
        if (d->count()) EXPECT_EQ(d->value(), val);

        return true;
    };

    EXPECT_FALSE( test({ "-a"                   }, ""       ) ); // -a expects an argument
    EXPECT_FALSE( test({ "-a", "xxx"            }, ""       ) ); // -a expects an argument
    EXPECT_TRUE ( test({ "-axxx"                }, "xxx"    ) );
    EXPECT_TRUE ( test({ "-a=xxx"               }, "=xxx"   ) );
    EXPECT_TRUE ( test({ "-b"                   }, ""       ) );
    EXPECT_FALSE( test({ "-b", "xxx"            }, ""       ) ); // unhandled positional xxx
    EXPECT_TRUE ( test({ "-bxxx"                }, "xxx"    ) );
    EXPECT_TRUE ( test({ "-b=xxx"               }, "=xxx"   ) );
    EXPECT_FALSE( test({ "-c"                   }, ""       ) ); // -c expects an argument
    EXPECT_TRUE ( test({ "-c", "xxx"            }, "xxx"    ) );
    EXPECT_FALSE( test({ "-cxxx"                }, ""       ) ); // unknown option -cxxx
    EXPECT_TRUE ( test({ "-c=xxx"               }, "xxx"    ) );
    EXPECT_TRUE ( test({ "-d"                   }, ""       ) );
    EXPECT_FALSE( test({ "-d", "xxx"            }, "xxx"    ) ); // unhandled positional xxx
    EXPECT_FALSE( test({ "-dxxx"                }, ""       ) ); // unknown option -dxxx
    EXPECT_TRUE ( test({ "-d=xxx"               }, "xxx"    ) );
}

TEST(CmdLineTest, Consume1)
{
    auto test = [](Argv const& argv, std::string const& s_val, std::vector<std::string> const& x_val) -> bool
    {
        SCOPED_TRACE("parsing: " + to_pretty_string(argv));

        cl::CmdLine cmd;

        auto a = cl::makeOption<std::string>(cl::Parser<>(), cmd, "a");
        auto s = cl::makeOption<std::string>(cl::Parser<>(), cmd, "script", cl::Positional, cl::Required, cl::ConsumeAfter);
        auto x = cl::makeOption<std::vector<std::string>>(cl::Parser<>(), cmd, "arguments", cl::Positional);

        if (!parse(cmd, argv))
            return false;

        if (s->count())
            EXPECT_EQ(s->value(), s_val);

        if (x->count())
            EXPECT_EQ(x->value(), x_val);
        else
            EXPECT_TRUE(x_val.empty());

        return true;
    };

    EXPECT_FALSE( test( { "-a"                      }, "script",    {           } ) ); // script name missing
    EXPECT_TRUE ( test( { "script"                  }, "script",    {           } ) );
    EXPECT_TRUE ( test( { "script", "x"             }, "script",    {"x"        } ) );
    EXPECT_TRUE ( test( { "x", "script"             }, "x",         {"script"   } ) );
    EXPECT_TRUE ( test( { "script", "-a"            }, "script",    {"-a"       } ) );
    EXPECT_TRUE ( test( { "-a", "script"            }, "script",    {           } ) ); // -a is an argument for <program>
    EXPECT_TRUE ( test( { "-a", "script", "-a"      }, "script",    {"-a"       } ) ); // the second -a does not match the "consume-option"
    EXPECT_TRUE ( test( { "-a", "script", "x", "-a" }, "script",    {"x", "-a"  } ) ); // the first -a is an argument for <program>
    EXPECT_TRUE ( test( { "script", "-a", "x"       }, "script",    {"-a", "x"  } ) );
    EXPECT_TRUE ( test( { "script", "x", "-a"       }, "script",    {"x", "-a"  } ) ); // -a is an argument for <s>
}

TEST(CmdLineTest, Consume2)
{
    // same as Consume1, but
    // merge script name and arguments...

    auto test = [](Argv const& argv, std::vector<std::string> const& s_val)
    {
        SCOPED_TRACE("parsing: " + to_pretty_string(argv));

        cl::CmdLine cmd;

        auto a = cl::makeOption<std::string>(cl::Parser<>(), cmd, "a");
        auto s = cl::makeOption<std::vector<std::string>>(cl::Parser<>(), cmd, "script", cl::Positional, cl::OneOrMore, cl::ConsumeAfter);

        if (!parse(cmd, argv))
            return false;

        if (s->count())
            EXPECT_EQ(s->value(), s_val);

        return true;
    };

    EXPECT_FALSE( test( { "-a"                      }, {                        } ) ); // script name missing
    EXPECT_TRUE ( test( { "script"                  }, {"script",               } ) );
    EXPECT_TRUE ( test( { "script", "x"             }, {"script", "x"           } ) );
    EXPECT_TRUE ( test( { "x", "script"             }, {"x",      "script"      } ) );
    EXPECT_TRUE ( test( { "script", "-a"            }, {"script", "-a"          } ) );
    EXPECT_TRUE ( test( { "-a", "script"            }, {"script",               } ) ); // -a is an argument for <program>
    EXPECT_TRUE ( test( { "-a", "script", "-a"      }, {"script", "-a"          } ) ); // the second -a does not match the "consume-option"
    EXPECT_TRUE ( test( { "-a", "script", "x", "-a" }, {"script", "x", "-a"     } ) ); // the first -a is an argument for <program>
    EXPECT_TRUE ( test( { "script", "-a", "x"       }, {"script", "-a", "x"     } ) );
    EXPECT_TRUE ( test( { "script", "x", "-a"       }, {"script", "x", "-a"     } ) ); // -a is an argument for <s>
}

TEST(CmdLineTest, Map1)
{
    auto test = [](bool result, Argv const& argv, std::pair<unsigned, int> x_val)
    {
        SCOPED_TRACE("parsing: " + to_pretty_string(argv));

        cl::CmdLine cmd;

        auto xParser = cl::MapParser<int>({
            { "none", 0 },
            { "c",    1 },
            { "c++",  2 },
            { "01234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789", 3}
        });

        auto x = cl::makeOption<int>(
            xParser,
            cmd, "x",
            cl::ArgRequired,
            cl::ArgName("lang"),
            cl::ZeroOrMore,
            cl::init(0)
            );

        bool actual_result = parse(cmd, argv);
        EXPECT_EQ(result, actual_result);

        EXPECT_EQ(x_val.first, x->count());

        if (x->count())
            EXPECT_EQ(x_val.second, x->value());
    };

    EXPECT_NO_FATAL_FAILURE( test(true,  {                  }, {0,0}) );
    EXPECT_NO_FATAL_FAILURE( test(false, {"-x"              }, {0,0}) );
    EXPECT_NO_FATAL_FAILURE( test(true,  {"-x", "none"      }, {1,0}) );
    EXPECT_NO_FATAL_FAILURE( test(true,  {"-x=none"         }, {1,0}) );
    EXPECT_NO_FATAL_FAILURE( test(true,  {"-x", "c++"       }, {1,2}) );
    EXPECT_NO_FATAL_FAILURE( test(true,  {"-x=c++"          }, {1,2}) );
    EXPECT_NO_FATAL_FAILURE( test(false, {"-x", "cxx"       }, {0,0}) );
    EXPECT_NO_FATAL_FAILURE( test(false, {"-x=cxx"          }, {0,0}) );
}

TEST(CmdLineTest, Map2)
{
    auto test = [](bool result, Argv const& argv, std::pair<unsigned, int> x_val)
    {
        SCOPED_TRACE("parsing: " + to_pretty_string(argv));

        cl::CmdLine cmd;

        auto xParser = cl::MapParser<int>({
            { "O0", 0 },
            { "O1", 1 },
            { "O2", 2 },
            { "O3", 3 },
        });

        auto x = cl::makeOption<int>(
            xParser,
            cmd,
            cl::Required,
            cl::ArgDisallowed
            );

        bool actual_result = parse(cmd, argv);
        EXPECT_EQ(result, actual_result);

        EXPECT_EQ(x_val.first, x->count());

        if (x->count())
            EXPECT_EQ(x_val.second, x->value());
    };

    EXPECT_NO_FATAL_FAILURE( test(false, {                  }, {0,0}) );
    EXPECT_NO_FATAL_FAILURE( test(false, {"-O"              }, {0,0}) );
    EXPECT_NO_FATAL_FAILURE( test(true,  {"-O1"             }, {1,1}) );
    EXPECT_NO_FATAL_FAILURE( test(false, {"-Ox"             }, {0,0}) );
    EXPECT_NO_FATAL_FAILURE( test(false, {"-O=1"            }, {0,0}) );
    EXPECT_NO_FATAL_FAILURE( test(false, {"-O", "1"         }, {0,0}) );
    EXPECT_NO_FATAL_FAILURE( test(false, {"-O1", "-O1"      }, {1,1}) );
    EXPECT_NO_FATAL_FAILURE( test(false, {"-O2", "-O1"      }, {1,2}) );
}

TEST(CmdLineTest, Map3)
{
    auto test = [](bool result, Argv const& argv, std::pair<unsigned, int> x_val)
    {
        SCOPED_TRACE("parsing: " + to_pretty_string(argv));

        cl::CmdLine cmd;

        auto xParser = cl::MapParser<int>({
            { "O0", 0 },
            { "O1", 1 },
            { "O2", 2 },
            { "O3", 3 },
        });

        auto x = cl::makeOption<int>(
            xParser,
            cmd,
            cl::Required,
            cl::Prefix,
            cl::ArgOptional
            );

        bool actual_result = parse(cmd, argv);
        EXPECT_EQ(result, actual_result);

        EXPECT_EQ(x_val.first, x->count());

        if (x->count())
            EXPECT_EQ(x_val.second, x->value());
    };

    EXPECT_NO_FATAL_FAILURE( test(false, {                  }, {0,0}) );
    EXPECT_NO_FATAL_FAILURE( test(false, {"-O"              }, {0,0}) );
    EXPECT_NO_FATAL_FAILURE( test(true,  {"-O1"             }, {1,1}) );
    EXPECT_NO_FATAL_FAILURE( test(false, {"-O1=O1"          }, {0,0}) );
    EXPECT_NO_FATAL_FAILURE( test(true,  {"-O1O1"           }, {1,1}) );
    EXPECT_NO_FATAL_FAILURE( test(true,  {"-O1O2"           }, {1,2}) );
    EXPECT_NO_FATAL_FAILURE( test(false, {"-O1Ox"           }, {0,0}) );
}

TEST(CmdLineTest, Map4)
{
    auto test = [](bool result, Argv const& argv, std::pair<unsigned, int> x_val)
    {
        SCOPED_TRACE("parsing: " + to_pretty_string(argv));

        cl::CmdLine cmd;

        auto xParser = cl::MapParser<int>({
            { "0", 0 },
            { "1", 1 },
            { "2", 2 },
            { "3", 3 },
        });

        auto x = cl::makeOption<int>(
            xParser,
            cmd, "O",
            cl::Required,
            cl::Prefix,
            cl::ArgRequired
            );

        bool actual_result = parse(cmd, argv);
        EXPECT_EQ(result, actual_result);

        EXPECT_EQ(x_val.first, x->count());

        if (x->count())
            EXPECT_EQ(x_val.second, x->value());
    };

    EXPECT_NO_FATAL_FAILURE( test(false, {                  }, {0,0}) );
    EXPECT_NO_FATAL_FAILURE( test(false, {"-O"              }, {0,0}) );
    EXPECT_NO_FATAL_FAILURE( test(true,  {"-O1"             }, {1,1}) );
    EXPECT_NO_FATAL_FAILURE( test(false, {"-Ox"             }, {0,0}) );
    EXPECT_NO_FATAL_FAILURE( test(false, {"-O=1"            }, {0,0}) );
    EXPECT_NO_FATAL_FAILURE( test(false, {"-O", "1"         }, {0,0}) );
    EXPECT_NO_FATAL_FAILURE( test(false, {"-O1", "-O1"      }, {1,1}) );
    EXPECT_NO_FATAL_FAILURE( test(false, {"-O2", "-O1"      }, {1,2}) );
}

TEST(CmdLineTest, MapRef)
{
    int opt = 0;

    cl::CmdLine cmd;

    auto x = cl::makeOption<int&>({{"0",0}, {"1",1}, {"2",2}}, cmd, cl::init(opt), "opt");

    bool ok = parse(cmd, {"-opt=1"});

    EXPECT_EQ(true, ok);
    EXPECT_EQ(1, opt);
}

TEST(CmdLineTest, Bump)
{
    cl::CmdLine cmd;

    auto atoi = [](StringRef str)
    {
        // using .data() is ok here...
        return str.data() ? std::atoi(str.data()) : 0;
    };

    auto parser = [&](int& value)
    {
        value += atoi(cmd.bump());
        value += atoi(cmd.bump());
        value += atoi(cmd.bump());
    };

    auto x = cl::makeOption<int>(
//        parser,
        std::bind(parser, std::placeholders::_3),
        cmd, "x", cl::init(0), cl::ArgOptional, cl::ZeroOrMore
        );

    bool ok = parse(cmd, {"-x", "1", "2", "3", "-x", "4"});
//    bool ok = parse(cmd, {"-x", "1", "xxxx", "3", "-x", "4"});

    EXPECT_EQ(true, ok);
    EXPECT_EQ(10, x->value());
}

TEST(CmdLineTest, AllowedValues)
{
    auto test = []()
    {
        cl::CmdLine cmd;

        enum OptimizationLevel {
            OL_None,
            OL_Trivial,
            OL_Default,
            OL_Expensive
        };

        auto optParser = cl::MapParser<OptimizationLevel>({
            { "O0", OL_None      },
            { "O1", OL_Trivial   },
            { "O2", OL_Default   },
            { "O3", OL_Expensive },
        });

        auto opt = cl::makeOption<OptimizationLevel>(
            std::ref(optParser),
            cmd,
            cl::ArgDisallowed,
            cl::ArgName("optimization level"),
            cl::init(OL_None),
            cl::Required
            );
    };

    EXPECT_NO_FATAL_FAILURE(test());
}
