// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_COMMON_LOCK_FREE_QUEUE_H
#define VSNRAY_COMMON_LOCK_FREE_QUEUE_H 1

#include <atomic>
#include <utility>

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Lock-free queue based on:
// http://www.drdobbs.com/parallel/writing-lock-free-code-a-corrected-queue/210604448
//

template <typename T>
class lock_free_queue
{
public:
    lock_free_queue()
        : first_(new node(T()))
    {
        last_.store(first_);
        separator_.store(first_);
    }

   ~lock_free_queue()
    {
        while (first_ != nullptr)
        {
            node* tmp = first_;
            first_ = tmp->next;
            delete tmp;
        }
    }

    void push_back(T&& value)
    {
        (*last_).next = new node(std::move(value));
        last_ = (*last_).next;

        while (first_ != separator_)
        {
            node* tmp = first_;
            first_ = first_->next;
            delete tmp;
        }
    }

    bool pop_front(T& value)
    {
        if (separator_ != last_)
        {
            value = std::move((*separator_).next->value);
            separator_ = (*separator_).next;
            return true;
        }
        else
        {
            // Queue is empty
            return false;
        }
    }

private:
    struct node
    {
        node(T&& v) : value(std::move(v)) {}

        T value;
        node* next = nullptr;
    };

    node* first_;
    std::atomic<node*> last_;
    std::atomic<node*> separator_;
};

} // visionaray

#endif // VSNRAY_COMMON_LOCK_FREE_QUEUE_H
