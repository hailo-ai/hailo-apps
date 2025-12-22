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
 * @ file example_common.h
 * Common macros and defines used by Hailort Examples
 **/

#ifndef _EXAMPLE_COMMON_H_
#define _EXAMPLE_COMMON_H_

#include <stdio.h>
#include <stdlib.h>
#include "hailo/hailort.h"
#include <queue>


#define RESET "\033[0m"
#define BLACK "\033[30m"              /* Black */
#define RED "\033[31m"                /* Red */
#define GREEN "\033[32m"              /* Green */
#define YELLOW "\033[33m"             /* Yellow */
#define BLUE "\033[34m"               /* Blue */
#define MAGENTA "\033[35m"            /* Magenta */
#define CYAN "\033[36m"               /* Cyan */
#define WHITE "\033[37m"              /* White */
#define BOLDBLACK "\033[1m\033[30m"   /* Bold Black */
#define BOLDRED "\033[1m\033[31m"     /* Bold Red */
#define BOLDGREEN "\033[1m\033[32m"   /* Bold Green */
#define BOLDYELLOW "\033[1m\033[33m"  /* Bold Yellow */
#define BOLDBLUE "\033[1m\033[34m"    /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m" /* Bold Magenta */
#define BOLDCYAN "\033[1m\033[36m"    /* Bold Cyan */
#define BOLDWHITE "\033[1m\033[37m"   /* Bold White */


template <typename T> 
class TSQueue { 
	private: 
		// Underlying queue 
		std::queue<T> m_queue; 
		// mutex for thread synchronization 
		std::mutex m_mutex; 
		// Condition variable for signaling 
		std::condition_variable m_cond; 
  
	public: 
		// Pushes an element to the queue 
		void push(T item) { 
			// Acquire lock 
			std::unique_lock<std::mutex> lock(m_mutex); 
			// Add item 
			m_queue.push(item); 
			// Notify one thread that 
			// is waiting 
			m_cond.notify_one(); 
		} 
	
		// Pops an element off the queue 
		T pop() { 
			// acquire lock 
			std::unique_lock<std::mutex> lock(m_mutex); 
			// wait until queue is not empty 
			m_cond.wait(lock, 
						[this]() { return !m_queue.empty(); }); 
	
			// retrieve item 
			T item = m_queue.front(); 
			m_queue.pop(); 
			// return item 
			return item; 
		}

		bool empty(){
			return m_queue.empty();
		}

		size_t size(){
			return m_queue.size();
		}  
}; 

#endif /* _EXAMPLE_COMMON_H_ */
