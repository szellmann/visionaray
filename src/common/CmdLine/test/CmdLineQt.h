// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

namespace support
{
namespace cl
{

#ifdef QSTRING_H
template <>
struct Parser<QString>
{
    void operator()(StringRef /*name*/, StringRef arg, QString& value) const
    {
        value.fromUtf8(arg.data(), arg.size());
    }
};
#endif

#ifdef QURL_H
template <>
struct Parser<QUrl>
{
    void operator()(StringRef /*name*/, StringRef arg, QUrl& value) const
    {
        value.setUrl(QString::fromUtf8(arg.data(), arg.size()), QUrl::StrictMode);

        if (!value.isValid())
        {
            throw std::runtime_error("invalid argument '" + arg + "' for option '" + name
                + "' (invalid URL)");
        }
    }
};
#endif

namespace qt
{
#ifdef QMAP_H
    struct QMapInserter
    {
        template <class C, class V>
        void operator()(C& c, V&& v) const {
            c.insert(std::forward<V>(v).first, std::forward<V>(v).second);
        }
    };
#endif

#ifdef QSET_H
    struct QSetInserter
    {
        template <class C, class V>
        void operator()(C& c, V&& v) const {
            c.insert(std::forward<V>(v));
        }
    };
#endif
}

#ifdef QHASH_H
template <class T, class U>
struct Traits<QHash<T, U>> : BasicTraits<std::pair<T, U>, qt::QMapInserter>
{
};

template <class T, class U>
struct Traits<QMultiHash<T, U>> : BasicTraits<std::pair<T, U>, qt::QMapInserter>
{
};
#endif

#ifdef QMAP_H
template <class T, class U>
struct Traits<QMap<T, U>> : BasicTraits<std::pair<T, U>, qt::QMapInserter>
{
};

template <class T, class U>
struct Traits<QMultiMap<T, U>> : BasicTraits<std::pair<T, U>, qt::QMapInserter>
{
};
#endif

#ifdef QSET_H
template <class T>
struct Traits<QSet<T>> : BasicTraits<typename QSet<T>::value_type, qt::QSetInserter>
{
};
#endif

#ifdef QSTRING_H
template <>
struct Traits<QString> : BasicTraits<QString, void>
{
};
#endif

} // namespace cl
} // namespace support
