// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include "Support/StringRef.h"
#include "Support/StringRefStream.h"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <vector>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4512) // assignment operator could not be generated
#endif

namespace support
{
namespace cl
{

//--------------------------------------------------------------------------------------------------
// Option flags
//

// Flags for the number of occurrences allowed
enum NumOccurrences : unsigned char {
    Optional,               // Zero or one occurrence allowed
    ZeroOrMore,             // Zero or more occurrences allowed
    Required,               // Exactly one occurrence required
    OneOrMore,              // One or more occurrences required
};

// Is a value required for the option?
enum NumArgs : unsigned char {
    ArgOptional,            // A value can appear... or not
    ArgRequired,            // A value is required to appear!
    ArgDisallowed,          // A value may not be specified (for flags)
};

// This controls special features that the option might have that cause it to be
// parsed differently...
enum Formatting : unsigned char {
    DefaultFormatting,      // Nothing special
    Prefix,                 // Must this option directly prefix its value?
    MayPrefix,              // Can this option directly prefix its value?
    Grouping,               // Can this option group with other options?
    Positional,             // Is a positional argument, no '-' required
};

enum MiscFlags : unsigned char {
    None                    = 0,
    CommaSeparated          = 0x01, // Should this list split between commas?
    ConsumeAfter            = 0x02, // Handle all following arguments as positional arguments
    Hidden                  = 0x04, // Do not show this option in the help message
};

//--------------------------------------------------------------------------------------------------
// CmdLine
//

class OptionBase;

class CmdLine
{
public:
    using OptionMap         = std::vector<std::pair<StringRef, OptionBase*>>;
    using OptionVector      = std::vector<OptionBase*>;
    using ConstOptionVector = std::vector<OptionBase const*>;
    using StringVector      = std::vector<std::string>;

private:
    // The current argument
    StringVector::const_iterator argCurr_;
    // End of command line arguments
    StringVector::const_iterator argLast_;
    // Index of the currently processed argument
    size_t index_;
    // List of options
    OptionMap options_;
    // List of positional options
    OptionVector positionals_;
    // The length of the longest prefix option
    size_t maxPrefixLength_;

public:
    // Constructor.
    CmdLine();

    // Destructor.
    ~CmdLine();

    // Adds the given option to the command line
    void add(OptionBase& opt);

    // Parse the given command line arguments
    void parse(StringVector const& argv, bool checkRequired = true);

    // Returns whether all command line arguments have been processed
    bool empty() const;

    // Returns the index of the currently processed argument
    size_t index() const;

    // Returns the current command line argument
    StringRef curr() const;

    // Returns the next argument and increments the index
    StringRef bump();

    // Returns a short usage description
    std::string usage() const;

    // Returns the help message
    std::string help(StringRef programName, StringRef overview = "") const;

private:
    void parse(bool checkRequired);

    OptionBase* findOption(StringRef name) const;

    ConstOptionVector getUniqueOptions() const;

    void handleArg(bool& dashdash, OptionVector::iterator& pos);

    void handlePositional(StringRef curr, OptionVector::iterator& pos);
    bool handleOption(StringRef curr);
    bool handlePrefix(StringRef curr);
    bool handleGroup(StringRef curr);

    void addOccurrence(OptionBase* opt, StringRef name);
    void addOccurrence(OptionBase* opt, StringRef name, StringRef arg);

    void parse(OptionBase* opt, StringRef name, StringRef arg);

    void check(OptionBase const* opt);
    void check();
};

//--------------------------------------------------------------------------------------------------
// ArgName
//

struct ArgName
{
    std::string value;

    explicit ArgName(std::string value) : value(std::move(value)) {}
};

//--------------------------------------------------------------------------------------------------
// Desc
//

struct Desc
{
    std::string value;

    explicit Desc(std::string value) : value(std::move(value)) {}
};

//--------------------------------------------------------------------------------------------------
// Initializer
//

namespace details
{
    template <class T>
    struct Initializer
    {
        T value;

        explicit Initializer(T x) : value(std::forward<T>(x)) {}

        // extract
        operator T() { return std::forward<T>(value); }
    };
}

template <class T>
inline auto init(T&& value) -> details::Initializer<T&&>
{
    return details::Initializer<T&&>(std::forward<T>(value));
}

//--------------------------------------------------------------------------------------------------
// Parser
//

template <class T = void>
struct Parser
{
    void operator()(StringRef name, StringRef arg, T& value) const
    {
        StringRefStream stream(arg);

        stream.setf(std::ios_base::fmtflags(0), std::ios::basefield);

        if (!(stream >> value) || !stream.eof())
            throw std::runtime_error("invalid argument '" + arg + "' for option '" + name + "'");
    }
};

template <>
struct Parser<bool>
{
    void operator()(StringRef name, StringRef arg, bool& value) const
    {
        if (arg.empty() || arg == "1" || arg == "true" || arg == "on")
            value = true;
        else if (arg == "0" || arg == "false" || arg == "off")
            value = false;
        else
            throw std::runtime_error("invalid argument '" + arg + "' for option '" + name + "'");
    }
};

template <>
struct Parser<std::string>
{
    void operator()(StringRef /*name*/, StringRef arg, std::string& value) const {
        value.assign(arg.data(), arg.size());
    }
};

template <>
struct Parser<void> // default parser
{
    template <class T>
    void operator()(StringRef name, StringRef arg, T& value) const {
        Parser<T>()(name, arg, value);
    }
};

//--------------------------------------------------------------------------------------------------
// MapParser
//

template <class T>
struct MapParser
{
    using ValueType = typename std::remove_reference<T>::type;

    struct MapValueType
    {
        // Key
        std::string key;
        // Value
        ValueType value;
        // An optional description of the value
        std::string desc;

        MapValueType(std::string key, ValueType value, std::string desc = "<< description missing >>")
            : key(std::move(key))
            , value(std::move(value))
            , desc(std::move(desc))
        {
        }
    };

    using MapType = std::vector<MapValueType>;

    MapType map;

    explicit MapParser(std::initializer_list<MapValueType> ilist)
        : map(ilist)
    {
    }

    void operator()(StringRef name, StringRef arg, ValueType& value) const
    {
        // If the arg is empty, the option is specified by name
        auto key = arg.empty() ? name : arg;

        auto I = std::find_if(map.begin(), map.end(),
            [&](MapValueType const& s) { return s.key == key; });

        if (I == map.end())
            throw std::runtime_error("invalid argument '" + arg + "' for option '" + name + "'");

        value = I->value;
    }

    std::vector<StringRef> getAllowedValues() const
    {
        std::vector<StringRef> vec;

        for (auto&& I : map)
            vec.emplace_back(I.key);

        return vec;
    }

    std::vector<StringRef> getDescriptions() const
    {
        std::vector<StringRef> vec;

        for (auto&& I : map)
            vec.emplace_back(I.desc);

        return vec;
    }
};

//--------------------------------------------------------------------------------------------------
// Traits
//

template <class ElementT, class InserterT = void>
struct BasicTraits
{
    using ElementType = ElementT;
    using InserterType = InserterT;
};

template <class T>
using ScalarType = BasicTraits<T>;

namespace details
{
    struct R2 {};
    struct R1 : R2 {};

    template <class T>
    struct RecRemoveCV {
        using type = typename std::remove_cv<T>::type;
    };

    template <template <class...> class T, class... A>
    struct RecRemoveCV<T<A...>> {
        using type = T<typename RecRemoveCV<A>::type...>;
    };

    template <class T> struct UnwrapReferenceWrapper { using type = T; };
    template <class T> struct UnwrapReferenceWrapper<std::reference_wrapper<T>> { using type = T; };

    struct Inserter
    {
        template <class C, class V>
        void operator()(C& c, V&& v) const {
            c.insert(c.end(), std::forward<V>(v));
        }
    };

    template <class T>
    auto TestInsert(R1) -> BasicTraits<typename RecRemoveCV<typename T::value_type>::type, Inserter>;

    template <class T>
    auto TestInsert(R2) -> BasicTraits<T>;
}

template <class T>
struct Traits : decltype(details::TestInsert<T>(details::R1()))
{
};

template <class T> struct Traits<T&> : Traits<T> {};
template <class T> struct Traits<std::reference_wrapper<T>> : Traits<T> {};

template <>
struct Traits<std::string> : BasicTraits<std::string>
{
};

//--------------------------------------------------------------------------------------------------
// OptionBase
//

class OptionBase
{
    friend class CmdLine;

    // The name of this option
    std::string name_;
    // The name of the value of this option
    std::string argName_;
    // Option description
    std::string desc_;
    // Controls how often the option must/may be specified on the command line
    NumOccurrences numOccurrences_;
    // Controls whether the option expects a value
    NumArgs numArgs_;
    // Controls how the option might be specified
    Formatting formatting_;
    // Other flags
    MiscFlags miscFlags_;
    // The number of times this option was specified on the command line
    unsigned count_;

protected:
    // Constructor.
    OptionBase();

public:
    // Destructor.
    virtual ~OptionBase();

    // Returns the name of this option
    std::string const& name() const { return name_; }

    // Return name of the value
    std::string const& argName() const { return argName_; }

    // Returns the option description
    std::string const& desc() const { return desc_; }

    // Returns the number of times this option has been specified on the command line
    unsigned count() const { return count_; }

    // Returns a short usage description
    std::string usage() const;

    // Returns the help message
    std::string help() const;

protected:
    void apply(std::string x)       { name_ = std::move(x); }
    void apply(ArgName x)           { argName_ = std::move(x.value); }
    void apply(Desc x)              { desc_ = std::move(x.value); }
    void apply(NumOccurrences x)    { numOccurrences_ = x; }
    void apply(NumArgs x)           { numArgs_ = x; }
    void apply(Formatting x)        { formatting_ = x; }
    void apply(MiscFlags x)         { miscFlags_ = static_cast<MiscFlags>(miscFlags_ | x); }

    template <class U>
    void apply(details::Initializer<U>) {}

    void applyAll() {}

    template <class A, class... Args>
    void applyAll(A&& a, Args&&... args)
    {
        apply(std::forward<A>(a));
        applyAll(std::forward<Args>(args)...);
    }

    template <class... Args>
    void applyAll(CmdLine& cmd, Args&&... args)
    {
        applyAll(std::forward<Args>(args)...);

        cmd.add(*this);
    }

private:
    StringRef displayName() const;

    bool isOccurrenceAllowed() const;
    bool isOccurrenceRequired() const;
    bool isUnbounded() const;
    bool isRequired() const;
    bool isPrefix() const;

    // Parses the given value and stores the result.
    virtual void parse(StringRef name, StringRef arg) = 0;

    // Returns a list of allowed values for this option
    virtual std::vector<StringRef> getAllowedValues() const = 0;

    // Returns a list of descriptions for the values this option accepts
    virtual std::vector<StringRef> getDescriptions() const = 0;
};

//--------------------------------------------------------------------------------------------------
// BasicOption<T>
//

template <class T>
class BasicOption : public OptionBase
{
    T value_;

protected:
    BasicOption(std::piecewise_construct_t) : value_() {}

    template <class U, class... Args>
    BasicOption(std::piecewise_construct_t, details::Initializer<U> x, Args&&...)
        : OptionBase()
        , value_(x)
    {
    }

    template <class A, class... Args>
    BasicOption(std::piecewise_construct_t, A&&, Args&&... args)
        : BasicOption(std::piecewise_construct, std::forward<Args>(args)...)
    {
    }

public:
    using ValueType = typename std::remove_reference<T>::type;

    // Returns the option value
    ValueType& value() { return value_; }

    // Returns the option value
    ValueType const& value() const { return value_; }
};

//--------------------------------------------------------------------------------------------------
// Option
//

template <class T, template <class> class TraitsT = Traits, class ParserT = Parser<typename TraitsT<T>::ElementType>>
class Option : public BasicOption<T>
{
public:
    using BaseType      = BasicOption<T>;
    using ElementType   = typename TraitsT<T>::ElementType;
    using InserterType  = typename TraitsT<T>::InserterType;
    using IsScalar      = typename std::is_void<InserterType>::type;

private:
    static_assert(IsScalar::value || std::is_default_constructible<ElementType>::value,
        "elements of containers must be default constructible");

    ParserT parser_;

public:
    using ParserType = typename details::UnwrapReferenceWrapper<ParserT>::type;

    template <class P, class... Args>
    explicit Option(std::piecewise_construct_t, P&& p, Args&&... args)
        : BaseType(std::piecewise_construct, std::forward<Args>(args)...)
        , parser_(std::forward<P>(p))
    {
        this->applyAll(IsScalar::value ? Optional : ZeroOrMore, std::forward<Args>(args)...);
    }

    // Returns the parser
    ParserType& parser() { return parser_; }

    // Returns the parser
    ParserType const& parser() const { return parser_; }

private:
    void parse(StringRef name, StringRef arg, std::false_type)
    {
        ElementType t;

        parser()(name, arg, t);

        InserterType()(this->value(), std::move(t));
    }

    void parse(StringRef name, StringRef arg, std::true_type) {
        parser()(name, arg, this->value());
    }

    virtual void parse(StringRef name, StringRef arg) override final {
        this->parse(name, arg, IsScalar());
    }

    template <class X = ParserType>
    auto getAllowedValues(details::R1) const -> decltype(std::declval<X const&>().getAllowedValues()) {
        return parser().getAllowedValues();
    }

    auto getAllowedValues(details::R2) const -> std::vector<StringRef> {
        return {};
    }

    virtual std::vector<StringRef> getAllowedValues() const override final {
        return this->getAllowedValues(details::R1());
    }

    template <class X = ParserType>
    auto getDescriptions(details::R1) const -> decltype(std::declval<X const&>().getDescriptions()) {
        return parser().getDescriptions();
    }

    auto getDescriptions(details::R2) const -> std::vector<StringRef> {
        return {};
    }

    virtual std::vector<StringRef> getDescriptions() const override final {
        return this->getDescriptions(details::R1());
    }
};

//--------------------------------------------------------------------------------------------------
// makeOption
//

// Construct a new Option, initialize the parser with the given value
template <class T, template <class> class TraitsT = Traits, class P, class... Args>
auto makeOption(P&& p, Args&&... args)
    -> std::unique_ptr<Option<T, TraitsT, typename std::decay<P>::type>>
{
    using U = Option<T, TraitsT, typename std::decay<P>::type>;

    return std::unique_ptr<U>(
        new U(std::piecewise_construct, std::forward<P>(p), std::forward<Args>(args)...));
}

// Construct a new Option, initialize the a map-parser with the given values
template <class T, template <class> class TraitsT = Traits, class... Args>
auto makeOption(std::initializer_list<typename MapParser<T>::MapValueType> ilist, Args&&... args)
    -> std::unique_ptr<Option<T, TraitsT, MapParser<T>>>
{
    using U = Option<T, TraitsT, MapParser<T>>;

    return std::unique_ptr<U>(
        new U(std::piecewise_construct, ilist, std::forward<Args>(args)...));
}

} // namespace cl
} // namespace support

#ifdef _MSC_VER
#pragma warning(pop)
#endif
