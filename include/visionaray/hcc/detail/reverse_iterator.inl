// This file is distributed under the MIT license.
// See the LICENSE file for details.

namespace visionaray
{
namespace hcc
{

//-------------------------------------------------------------------------------------------------
// reverse_iterator members
//

template <typename It>
VSNRAY_FUNC
reverse_iterator<It>::reverse_iterator(It x)
    : current(x)
{
}

template <typename It>
template <typename OtherIt>
VSNRAY_FUNC
reverse_iterator<It>::reverse_iterator(reverse_iterator<OtherIt> const& rhs)
    : reverse_iterator<It>(rhs)
{
}

template <typename It>
template <typename OtherIt>
VSNRAY_FUNC
reverse_iterator<It>& reverse_iterator<It>::operator=(reverse_iterator<OtherIt> const& rhs)
{
    if (&rhs != this)
    {
        *this = rhs;
    }
    return *this;
}

template <typename It>
VSNRAY_FUNC
It reverse_iterator<It>::base() const
{
    return current;
}

template <typename It>
VSNRAY_FUNC
typename reverse_iterator<It>::reference reverse_iterator<It>::operator*() const
{
    It tmp = current;
    return *--tmp;
}

template <typename It>
VSNRAY_FUNC
typename reverse_iterator<It>::pointer reverse_iterator<It>::operator->() const
{
    It tmp = current;
    return *--tmp;
}

template <typename It>
VSNRAY_FUNC
typename reverse_iterator<It>::value_type reverse_iterator<It>::operator[](
        typename reverse_iterator<It>::difference_type n
        ) const
{
    return current[-n - 1];
}

template <typename It>
VSNRAY_FUNC
reverse_iterator<It>& reverse_iterator<It>::operator++()
{
    --current;
    return *this;
}

template <typename It>
VSNRAY_FUNC
reverse_iterator<It> reverse_iterator<It>::operator++(int)
{
    It tmp = current;
    current--;
    return reverse_iterator(tmp);
}

template <typename It>
VSNRAY_FUNC
reverse_iterator<It>& reverse_iterator<It>::operator--()
{
    ++current;
    return *this;
}

template <typename It>
VSNRAY_FUNC
reverse_iterator<It> reverse_iterator<It>::operator--(int)
{
    It tmp = current;
    current++;
    return reverse_iterator(tmp);
}

template <typename It>
VSNRAY_FUNC
reverse_iterator<It> reverse_iterator<It>::operator+(
        typename reverse_iterator<It>::difference_type n
        )
{
    It tmp = current;
    current -= n;
    return reverse_iterator(tmp);
}

template <typename It>
VSNRAY_FUNC
reverse_iterator<It> reverse_iterator<It>::operator-(
        typename reverse_iterator<It>::difference_type n
        )
{
    It tmp = current;
    current + n;
    return reverse_iterator(tmp);
}

template <typename It>
VSNRAY_FUNC
reverse_iterator<It>& reverse_iterator<It>::operator+=(
        typename reverse_iterator<It>::difference_type n
        )
{
    current -= n;
    return *this;
}

template <typename It>
VSNRAY_FUNC
reverse_iterator<It>& reverse_iterator<It>::operator-=(
        typename reverse_iterator<It>::difference_type n
        )
{
    current += n;
    return *this;
}


//-------------------------------------------------------------------------------------------------
// Comparisons
//

template <typename It1, typename It2>
VSNRAY_FUNC
bool operator==(reverse_iterator<It1> const& a, reverse_iterator<It2> const& b)
{
    return a.base() == b.base();
}

template <typename It1, typename It2>
VSNRAY_FUNC
bool operator!=(reverse_iterator<It1> const& a, reverse_iterator<It2> const& b)
{
    return a.base() != b.base();
}

template <typename It1, typename It2>
VSNRAY_FUNC
bool operator<(reverse_iterator<It1> const& a, reverse_iterator<It2> const& b)
{
    return a.base() > b.base();
}

template <typename It1, typename It2>
VSNRAY_FUNC
bool operator>(reverse_iterator<It1> const& a, reverse_iterator<It2> const& b)
{
    return a.base() < b.base();
}

template <typename It1, typename It2>
VSNRAY_FUNC
bool operator<=(reverse_iterator<It1> const& a, reverse_iterator<It2> const& b)
{
    return a.base() >= b.base();
}

template <typename It1, typename It2>
VSNRAY_FUNC
bool operator>=(reverse_iterator<It1> const& a, reverse_iterator<It2> const& b)
{
    return a.base() <= b.base();
}


//-------------------------------------------------------------------------------------------------
// Arithmetic
//

template <typename It>
VSNRAY_FUNC
reverse_iterator<It> operator+(
        typename reverse_iterator<It>::difference_type n,
        reverse_iterator<It> const& it)
{
    return it + n;
}

template <typename It>
VSNRAY_FUNC
reverse_iterator<It> operator-(
        typename reverse_iterator<It>::difference_type n,
        reverse_iterator<It> const& it)
{
    return it - n;
}

} // hcc
} // visionaray
