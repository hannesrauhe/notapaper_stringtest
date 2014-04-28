/*
 * stringtest.h
 *
 *  Created on: Jan 21, 2013
 *      Author: hannes
 */

#ifndef STRINGTEST_H_
#define STRINGTEST_H_

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <limits>
#include <iostream>
#include <fstream>
#include <boost/timer/timer.hpp>
#include <omp.h>

//#define USE_MEMCMP 1

#ifdef __CDT_PARSER__
#define cudaHostAllocDefault 1
#endif

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {                                          \
    cudaError_t _m_cudaStat = value;                                        \
    if (_m_cudaStat != cudaSuccess) {                                       \
        fprintf(stderr, "Error %s at line %d in file %s\n",                 \
                cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);       \
        exit(1);                                                            \
    } }

struct gpuStringArray {
    unsigned size;
    unsigned byte_size;
    unsigned int * pos;
    unsigned int * length;  // could be a smaller type if strings are short
    char * first;
    char * data; // 32 bit data type will improve memory throughput, could be 8 bit

    void init(std::vector<std::string>& haystack, int noChar);
    void destroy();

    ///only works if sorted...
    int lower_bound( int first, int last, const char* value, int length ) const;
    int upper_bound( int first, int last, const char* value, int length ) const;

    const char* operator[](int entryno) const;

    static bool strcmplt_cpu(const char* str1, const char* str2, uint l1, uint l2);
    static bool strcmp2_cpu(const char* str1, const char* str2, uint length);
};

/***
 * dict encoded string array where dict is a gpuStringArray
 * valueIDs are stored in a simple vector of integers <-- not optimal, type should depend on maximum size (bitcompressed)
 */
class dict_haystack {
public:
    gpuStringArray _dict;
    std::vector<uint> _valueIDs;

    dict_haystack(const std::vector<std::string>& haystack);
    ~dict_haystack();
    const char* operator[](int pos);
};

struct timestruct {
	timestruct(float p = 0.0, float t = 0.0, float s = 0.0, float o = 0.0, float k = 0.0, int r=0) :
		partime(p),transfer(t),seqtime(s),overhead(o),kernel(k),run(r) {}
	float partime;
	float transfer;
	float seqtime;
	float overhead;
	float kernel;
	uint run;
	static const float tolerance = 50;
	std::string test_name;

	timestruct operator/(float div) {
		return timestruct(partime/div,transfer/div,seqtime/div,overhead/div,kernel/div);
	}
//	timestruct operator+(const timestruct &t) {
//		return timestruct(partime+t.partime,transfer+t.transfer,seqtime+t.seqtime,overhead+t.overhead,kernel+t.kernel,run+1);
//	}

	void compare_approx(float a, float b, const char* time_p) const {
		if(run && a!=0.0 && b>0.0) {
			float av = a/static_cast<float>(run);
			float a_begin = av - av*tolerance/100;
			float a_end = av + av*tolerance/100;
			if(b>a_end || b<a_begin) {
				std::cerr<<"High variance at run "<<run<<" of test "<<test_name<<"("<< time_p <<"), avg: "<<av<<", next value:"<<b<<" ("<< (b-av)/av*100.0 <<"%)"<<std::endl;
			}
		}
	}

	void compare_approx(const timestruct &t) const {
		if(t.run==0) {
			compare_approx(partime,t.partime,"partime");
			compare_approx(transfer,t.transfer,"transfer");
			compare_approx(seqtime,t.seqtime,"seqtime");
			compare_approx(overhead,t.overhead,"overhead");
			compare_approx(kernel,t.kernel,"kernel runtime");
		} else {
			std::cerr<<"compare_approx - given timestruct has already saved an average - don't build the average from those."<<std::endl;
			//this is not intended to happen <- if you want this, change the implementation accordingly
		}
	}

	timestruct& operator+=(const timestruct &t) {
		if(t.run>0) {
			std::cerr<<"addition - given timestruct already is an average - don't build the average from those."<<std::endl;
			//this is not intended to happen <- if you want this, change the implementation accordingly
		}
		compare_approx(t);
		partime+=t.partime;
		transfer+=t.transfer;
		seqtime+=t.seqtime;
		overhead+=t.overhead;
		kernel+=t.kernel;
		++run;
		return *this;
	}

	timestruct& set_name(std::string n) {
		test_name=n;
		return *this;
	}
};

std::ostream &operator<<(std::ostream &stream, const timestruct &t);

class time_printer {
public:
    time_printer(float total_time, std::ostream& outstream = std::cout) : total(total_time),os(outstream) {
        os<<"Total: "<<total_time/1000.0<<std::endl;
//        printf("%20s: %.5f\n","Total",total_time/1000.0);
    }

    void print(const char* str, float part_time) {
        os<<str<<": "<<part_time/1000.0<<" "<<part_time/total*100.0<<std::endl;
//        printf("%20s: %.5f %5.1f%%\n",str,part_time/1000.0,part_time/total*100.0);
    }

private:
    float total;
    std::ostream& os;
};

class stringtest_interface {
protected:
    std::ostream& _os;
    bool _measure_transfer;
public:
    stringtest_interface(std::ostream& outstream = std::cout, bool measure_transfer = true) : _os(outstream),_measure_transfer(measure_transfer) {}
    virtual ~stringtest_interface() {};

	virtual timestruct cpu_bench(std::vector<std::string>& haystack, const char* needle) { return timestruct(-1).set_name("not implemented"); }
	virtual timestruct cpu_bench_parallel(std::vector<std::string>& haystack, const char* needle) { return timestruct(-1).set_name("not implemented"); }

	virtual timestruct cpu_bench(const gpuStringArray& haystack, const char* needle) { return timestruct(-1).set_name("not implemented"); }
	virtual timestruct cpu_bench_parallel(const gpuStringArray& haystack, const char* needle) { return timestruct(-1).set_name("not implemented"); }
	virtual timestruct cpu_bench_with_index(const gpuStringArray& haystack, const char* needle) { return timestruct(-1).set_name("not implemented"); }
	virtual timestruct cpu_bench_with_index_SSE(const gpuStringArray& haystack, const char* needle) { return timestruct(-1).set_name("not implemented"); }
	virtual timestruct cpu_bench_parallel_with_index(const gpuStringArray& haystack, const char* needle) { return timestruct(-1).set_name("not implemented"); }
	virtual timestruct cpu_bench_parallel_with_index_SSE(const gpuStringArray& haystack, const char* needle) { return timestruct(-1).set_name("not implemented"); }
	virtual timestruct gpu_bench(const gpuStringArray& haystack, const char* needle, const int THREADS_PER_BLOCK = 256) { return timestruct(-1).set_name("not implemented"); }
	virtual timestruct gpu_bench_with_index(const gpuStringArray& haystack, const char* needle, const int THREADS_PER_BLOCK = 256) { return timestruct(-1).set_name("not implemented"); }

	virtual timestruct cpu_bench(const dict_haystack& haystack, const char* needle) { return timestruct(-1).set_name("not implemented"); }
	virtual timestruct cpu_bench_parallel(const dict_haystack& haystack, const char* needle) { return timestruct(-1).set_name("not implemented"); }
	virtual timestruct gpu_bench(const dict_haystack& haystack, const char* needle, const int THREADS_PER_BLOCK = 256) { return timestruct(-1).set_name("not implemented"); }
};

class startswith_test : public stringtest_interface{
public:
    startswith_test(std::ostream& outstream = std::cout, bool measure_transfer = true) : stringtest_interface(outstream, measure_transfer) {}

	timestruct cpu_bench(std::vector<std::string>& haystack, const char* needle);
	timestruct cpu_bench_parallel(std::vector<std::string>& haystack, const char* needle);

	timestruct cpu_bench(const gpuStringArray& haystack, const char* needle);
	timestruct cpu_bench_parallel(const gpuStringArray& haystack, const char* needle);
	timestruct cpu_bench_with_index(const gpuStringArray& haystack, const char* needle);
	timestruct cpu_bench_with_index_SSE(const gpuStringArray& haystack, const char* needle);
	timestruct cpu_bench_parallel_with_index(const gpuStringArray& haystack, const char* needle);
	timestruct cpu_bench_parallel_with_index_SSE(const gpuStringArray& haystack, const char* needle);
	timestruct gpu_bench(const gpuStringArray& haystack, const char* needle, const int THREADS_PER_BLOCK = 256);
	timestruct gpu_bench_with_index(const gpuStringArray& haystack, const char* needle, const int THREADS_PER_BLOCK = 256);

    timestruct cpu_bench(const dict_haystack& haystack, const char* needle);
    timestruct cpu_bench_parallel(const dict_haystack& haystack, const char* needle);
    timestruct gpu_bench(const dict_haystack& haystack, const char* needle, const int THREADS_PER_BLOCK = 256);
};


class substr_test : public stringtest_interface {
public:
    substr_test(std::ostream& outstream = std::cout, bool measure_transfer = true) : stringtest_interface(outstream, measure_transfer) {}

	timestruct cpu_bench(std::vector<std::string>& haystack, const char* needle);
	timestruct cpu_bench_parallel(std::vector<std::string>& haystack, const char* needle);

	timestruct cpu_bench(const gpuStringArray& haystack, const char* needle);
	timestruct cpu_bench_parallel(const gpuStringArray& haystack, const char* needle);
	timestruct gpu_bench(const gpuStringArray& haystack, const char* needle, const int THREADS_PER_BLOCK = 256);

    timestruct cpu_bench(const dict_haystack& haystack, const char* needle);
    timestruct cpu_bench_parallel(const dict_haystack& haystack, const char* needle);
//    timestruct gpu_bench(const dict_haystack& haystack, const char* needle, const int THREADS_PER_BLOCK = 256);
};

void gpureset();
#endif /* STRINGTEST_H_ */
