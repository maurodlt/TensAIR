/**
 * Copyright (c) 2020 University of Luxembourg. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are
 * permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this list of
 * conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice, this list
 * of conditions and the following disclaimer in the documentation and/or other materials
 * provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors may be
 * used to endorse or promote products derived from this software without specific prior
 * written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE UNIVERSITY OF LUXEMBOURG AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
 * THE UNIVERSITY OF LUXEMBOURG OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
 * OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
 * TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **/

/*
 * Partition.hpp
 *
 *  Created on: Aug 31, 2018
 *  Author: martin.theobald
 */

#ifndef PARTITIONING_PARTITION_HPP_
#define PARTITIONING_PARTITION_HPP_

#include <iostream>
#include <vector>

using namespace std;

template<typename T>
class Partition {

public:

	Partition() {
	}

	~Partition() {
	}

	T get(int idx) {
		return values[idx];
	}

	void set(T value, int idx) {
		values[idx] = value;
	}

	void add(T value) {
		values.push_back(value);

	}

	void clear() {
		values.clear();
	}

	int size() {
		return (int)(values.size());
	}

	void print() {
		typename vector<T>::iterator it = values.begin();
		int i = 0;
		while (it != values.end()) {
			cout << i << "\t VALUE [" << it->first << "]" << endl;
			it++;
			i++;
		}
	}

private:

	vector<T> values;

};

#endif /* PARTITIONING_PARTITION_HPP_ */
