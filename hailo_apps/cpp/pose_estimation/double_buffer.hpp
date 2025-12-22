/**
 * Copyright 2020 (C) Hailo Technologies Ltd.
 * All rights reserved.
 *
 * Hailo Technologies Ltd. ("Hailo") disclaims any warranties, including, but not limited to,
 * the implied warranties of merchantability and fitness for a particular purpose.
 * This software is provided on an "AS IS" basis, and Hailo has no obligation to provide maintenance,
 * support, updates, enhancements, or modifications.
 *
 * You may use this software in the development of any project.
 * You shall not reproduce, modify or distribute this software without prior written permission.
 **/
/**
 * @file double_buffer.hpp
 * @brief Implementation of DoubleBuffer class for detection app example
 **/

#ifndef _HAILO_DOUBLE_BUFFER_HPP_
#define _HAILO_DOUBLE_BUFFER_HPP_

#include <stdint.h>
#include <vector>
#include <mutex>
#include <condition_variable>

template <typename T>
class DoubleBuffer {
public:
    DoubleBuffer(uint32_t size) : m_first_buffer(size), m_second_buffer(size),
        m_write_ptr(&m_first_buffer), m_read_ptr(&m_first_buffer)
    {}

    std::vector<T> &get_write_buffer()
    {
        return m_write_ptr->acquire_write_buffer();
    }

    void release_write_buffer()
    {
        m_write_ptr->release_buffer();
        m_write_ptr = (m_write_ptr == &m_first_buffer) ? &m_second_buffer : &m_first_buffer;
    }

    std::vector<T> &get_read_buffer()
    {
        return m_read_ptr->acquire_read_buffer();
    }

    void release_read_buffer()
    {
        m_read_ptr->release_buffer();
        m_read_ptr = (m_read_ptr == &m_first_buffer) ? &m_second_buffer : &m_first_buffer;
    }

private:
    class SafeBuffer {
    public:
        SafeBuffer(uint32_t size) :
        m_state(State::WRITE), m_cv(), m_mutex(), m_buffer(size)
        {}

        std::vector<T> &acquire_write_buffer()
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_cv.wait(lock, [this]{ return (State::WRITE == m_state); });

            return m_buffer;
        }

        std::vector<T> &acquire_read_buffer()
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_cv.wait(lock, [this]{ return (State::READ == m_state); });

            return m_buffer;
        }

        void release_buffer()
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            swap_state();
            m_cv.notify_one();
        }

    private:
        void swap_state() {
            m_state = (State::WRITE == m_state) ? State::READ : State::WRITE;
        }

        enum class State {
            READ = 0,
            WRITE = 1,
        };

        State m_state;
        std::condition_variable m_cv;
        std::mutex m_mutex;
        std::vector<T> m_buffer;
    };

    SafeBuffer m_first_buffer;
    SafeBuffer m_second_buffer;

    SafeBuffer *m_read_ptr;
    SafeBuffer *m_write_ptr;
};

#endif /* _HAILO_DOUBLE_BUFFER_HPP_ */
