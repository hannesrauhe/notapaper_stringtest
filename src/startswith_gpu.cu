#include "stringtest.hpp"
#include "startswith_gpu_kernels.cuh"

//#define STRINGTEST_USE_PINNED_MEMORY
//#define STRINGTEST_USE_UNOPTIMIZED_KERNEL

__global__ void valueID_select_kernel(int* vids, uint size, int ent1, bool* result) {
    uint pos = blockDim.x*blockIdx.x + threadIdx.x;
    if(pos<size) {
        result[pos]=(vids[pos]==ent1);
    }
}

__global__ void valueID_select_kernel(int* vids, uint size, int ent1, int ent2, bool* result) {
    uint pos = blockDim.x*blockIdx.x + threadIdx.x;
    if(pos<size) {
        result[pos]=(vids[pos]>=ent1) && (vids[pos]<=ent2);
    }
}


//fermi device should use constant memory without this
//__constant__ char CONSTANT_NEEDLE[50];

__global__ void startswith_kernel(const gpuStringArray haystack, const char* needle, uint l, bool* result) {
    uint pos = blockDim.x*blockIdx.x + threadIdx.x;
    if(haystack.length[pos]>=l && pos<haystack.size) {
        result[pos]=strcmp2(&haystack.data[haystack.pos[pos]],needle,l);
    } else {
        result[pos]=false;
    }
}


timestruct startswith_test::gpu_bench(const gpuStringArray& haystack, const char* needle, const int THREADS_PER_BLOCK) {
    assert(strlen(needle)<50);
    int c = 0;
    float kernel_time = 0.0;
    float transfer_time = 0.0;
    float result_time = 0.0;
    float free_time = 0.0;
    float total_time = 0.0;
    float malloc_time = 0.0;
    float cpu_time = 0.0;
    boost::timer::cpu_timer t2;

    {
        boost::timer::cpu_timer t;

        char* dev_needle;
        gpuStringArray dev_strarr;
        dev_strarr.size = haystack.size;
        cudaEvent_t start, stop;

#ifdef STRINGTEST_USE_UNOPTIMIZED_KERNEL
        bool* result;
        bool* dev_result;
        uint noBlocks=haystack.size/THREADS_PER_BLOCK+1;
        uint noChars=haystack.pos[haystack.size-1]+haystack.length[haystack.size-1];

        cudaEventCreate(&start);
        cudaEventCreate(&stop);
#ifdef STRINGTEST_MEASURE_TRANSFER
        cudaEventRecord( start, 0 );
#endif
        CUDA_CHECK_RETURN(cudaMalloc((void**) &dev_strarr.data, sizeof(char) * noChars));
        CUDA_CHECK_RETURN(cudaMalloc((void**) &dev_strarr.length, sizeof(uint) * haystack.size));
        CUDA_CHECK_RETURN(cudaMalloc((void**) &dev_strarr.pos, sizeof(uint) * haystack.size));

#ifndef STRINGTEST_MEASURE_TRANSFER
        cudaEventRecord( start, 0 );
#endif
        CUDA_CHECK_RETURN(cudaMalloc((void**) &dev_result, sizeof(bool) * haystack.size));
        CUDA_CHECK_RETURN(cudaMalloc((void**) &dev_needle, sizeof(char) * strlen(needle)));
#ifdef STRINGTEST_USE_PINNED_MEMORY
        CUDA_CHECK_RETURN(cudaHostAlloc(&result,haystack.size*sizeof(bool),cudaHostAllocDefault));
#else
        result = (bool*)malloc(haystack.size*sizeof(bool));
#endif
        cudaEventRecord( stop, 0 );

        cudaEventSynchronize( stop );
        cudaEventElapsedTime( &malloc_time, start, stop );
        cudaEventDestroy( start );
        cudaEventDestroy( stop );

        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord( start, 0 );
#ifdef STRINGTEST_MEASURE_TRANSFER
        cudaEventRecord( start, 0 );
#endif
        CUDA_CHECK_RETURN(cudaMemcpy(dev_strarr.data, haystack.data, sizeof(char) * noChars, cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(cudaMemcpy(dev_strarr.length, haystack.length, sizeof(uint) * haystack.size, cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(cudaMemcpy(dev_strarr.pos, haystack.pos, sizeof(uint) * haystack.size, cudaMemcpyHostToDevice));
#ifndef STRINGTEST_MEASURE_TRANSFER
        cudaEventRecord( start, 0 );
#endif
        CUDA_CHECK_RETURN(cudaMemcpy(dev_needle,needle,strlen(needle),cudaMemcpyHostToDevice));
        cudaEventRecord( stop, 0 );

        cudaEventSynchronize( stop );
        cudaEventElapsedTime( &transfer_time, start, stop );
        cudaEventDestroy( start );
        cudaEventDestroy( stop );


        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord( start, 0 );
        startswith_kernel<<<noBlocks, THREADS_PER_BLOCK>>>(dev_strarr,dev_needle,strlen(needle),dev_result);
        cudaEventRecord( stop, 0 );

        cudaEventSynchronize( stop );
        cudaEventElapsedTime( &kernel_time, start, stop );
        cudaEventDestroy( start );
        cudaEventDestroy( stop );


        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord( start, 0 );
        CUDA_CHECK_RETURN(cudaMemcpy(result, dev_result, haystack.size*sizeof(bool), cudaMemcpyDeviceToHost));

        CUDA_CHECK_RETURN(cudaThreadSynchronize()); // Wait for the GPU launched work to complete
        CUDA_CHECK_RETURN(cudaGetLastError());
        cudaEventRecord( stop, 0 );

        cudaEventSynchronize( stop );
        cudaEventElapsedTime( &result_time, start, stop );
        cudaEventDestroy( start );
        cudaEventDestroy( stop );

        t2.start();
#pragma omp parallel for reduction(+:c)
        for (uint i = 0; i < haystack.size; i++)
           c+=(true==result[i]);
        t2.stop();

#else //optimized kernel version

        uint* result;
        uint* dev_result;
        uint noBlocks=haystack.size/THREADS_PER_BLOCK+1;
        uint noChars=haystack.pos[haystack.size-1]+haystack.length[haystack.size-1];
        int smemSize = (THREADS_PER_BLOCK <= 32) ? 2 * THREADS_PER_BLOCK * sizeof(uint) : THREADS_PER_BLOCK * sizeof(uint);

        t2.start();
        CUDA_CHECK_RETURN(cudaMalloc((void**) &dev_strarr.data, sizeof(char) * noChars));
        CUDA_CHECK_RETURN(cudaMalloc((void**) &dev_strarr.length, sizeof(uint) * haystack.size));
        CUDA_CHECK_RETURN(cudaMalloc((void**) &dev_strarr.pos, sizeof(uint) * haystack.size));
        t2.stop();
        if(_measure_transfer) {
            malloc_time = t2.elapsed().wall/1000000.0;
        }

        t2.start();
        CUDA_CHECK_RETURN(cudaMalloc((void**) &dev_result, sizeof(uint) * noBlocks));
        CUDA_CHECK_RETURN(cudaMalloc((void**) &dev_needle, sizeof(char) * strlen(needle)));
#ifdef STRINGTEST_USE_PINNED_MEMORY
        CUDA_CHECK_RETURN(cudaHostAlloc(&result,noBlocks*sizeof(uint),cudaHostAllocDefault));
#else
        result = (uint*)malloc(noBlocks*sizeof(uint));
#endif
        t2.stop();
        malloc_time += t2.elapsed().wall/1000000.0;


        t2.start();
        CUDA_CHECK_RETURN(cudaMemcpy(dev_strarr.data, haystack.data, sizeof(char) * noChars, cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(cudaMemcpy(dev_strarr.length, haystack.length, sizeof(uint) * haystack.size, cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(cudaMemcpy(dev_strarr.pos, haystack.pos, sizeof(uint) * haystack.size, cudaMemcpyHostToDevice));
        t2.stop();
        if(_measure_transfer) {
            transfer_time = t2.elapsed().wall/1000000.0;
        }
        t2.start();
        CUDA_CHECK_RETURN(cudaMemcpy(dev_needle,needle,strlen(needle),cudaMemcpyHostToDevice));
        t2.stop();
        transfer_time += t2.elapsed().wall/1000000.0;


        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord( start, 0 );
        switch (THREADS_PER_BLOCK)
        {
            case 512:
                startswith_kernel_opt<512, false><<< noBlocks, THREADS_PER_BLOCK, smemSize >>>(dev_strarr,dev_needle,strlen(needle),haystack.size,dev_result); break;
            case 256:
                startswith_kernel_opt<256, false><<< noBlocks, THREADS_PER_BLOCK, smemSize >>>(dev_strarr,dev_needle,strlen(needle),haystack.size,dev_result); break;
            case 128:
                startswith_kernel_opt<128, false><<< noBlocks, THREADS_PER_BLOCK, smemSize >>>(dev_strarr,dev_needle,strlen(needle),haystack.size,dev_result); break;
            case 64:
                startswith_kernel_opt< 64, false><<< noBlocks, THREADS_PER_BLOCK, smemSize >>>(dev_strarr,dev_needle,strlen(needle),haystack.size,dev_result); break;
            case 32:
                startswith_kernel_opt< 32, false><<< noBlocks, THREADS_PER_BLOCK, smemSize >>>(dev_strarr,dev_needle,strlen(needle),haystack.size,dev_result); break;
            case 16:
                startswith_kernel_opt< 16, false><<< noBlocks, THREADS_PER_BLOCK, smemSize >>>(dev_strarr,dev_needle,strlen(needle),haystack.size,dev_result); break;
            case  8:
                startswith_kernel_opt<  8, false><<< noBlocks, THREADS_PER_BLOCK, smemSize >>>(dev_strarr,dev_needle,strlen(needle),haystack.size,dev_result); break;
            case  4:
                startswith_kernel_opt<  4, false><<< noBlocks, THREADS_PER_BLOCK, smemSize >>>(dev_strarr,dev_needle,strlen(needle),haystack.size,dev_result); break;
            case  2:
                startswith_kernel_opt<  2, false><<< noBlocks, THREADS_PER_BLOCK, smemSize >>>(dev_strarr,dev_needle,strlen(needle),haystack.size,dev_result); break;
            case  1:
                startswith_kernel_opt<  1, false><<< noBlocks, THREADS_PER_BLOCK, smemSize >>>(dev_strarr,dev_needle,strlen(needle),haystack.size,dev_result); break;
        }
        cudaEventRecord( stop, 0 );

        cudaEventSynchronize( stop );
        cudaEventElapsedTime( &kernel_time, start, stop );
        cudaEventDestroy( start );
        cudaEventDestroy( stop );


        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord( start, 0 );
        CUDA_CHECK_RETURN(cudaMemcpy(result, dev_result, noBlocks*sizeof(uint), cudaMemcpyDeviceToHost));

        CUDA_CHECK_RETURN(cudaThreadSynchronize()); // Wait for the GPU launched work to complete
        CUDA_CHECK_RETURN(cudaGetLastError());
        cudaEventRecord( stop, 0 );

        cudaEventSynchronize( stop );
        cudaEventElapsedTime( &result_time, start, stop );
        cudaEventDestroy( start );
        cudaEventDestroy( stop );

        t2.start();
//#pragma omp parallel for reduction(+:c)
        for (uint i = 0; i < noBlocks; i++) {
           c+=result[i];
        }
        t2.stop();
#endif

        t2.start();
        CUDA_CHECK_RETURN(cudaFree(dev_strarr.data));
        CUDA_CHECK_RETURN(cudaFree(dev_strarr.length));
        CUDA_CHECK_RETURN(cudaFree(dev_strarr.pos));
        t2.stop();
        if(_measure_transfer) {
            free_time = t2.elapsed().wall/1000000.0;
        }

        t2.start();
        CUDA_CHECK_RETURN(cudaFree(dev_result));
        CUDA_CHECK_RETURN(cudaFree(dev_needle));
#ifdef STRINGTEST_USE_PINNED_MEMORY
        CUDA_CHECK_RETURN(cudaFreeHost(result));
#else
        free(result);
#endif

        t2.stop();
        free_time += t2.elapsed().wall/1000000.0;

        total_time = t.elapsed().wall;
    }

    total_time = total_time/1000000.0;

    _os<<"custom_string_arr on GPU:      "<<c<<" strings in haystack start with needle "<<needle<<std::endl;

    time_printer pr(total_time,_os);
    pr.print("alloc time", malloc_time);
    pr.print("transfer time",transfer_time);
    pr.print("kernel time",kernel_time);
    pr.print("result transfer",result_time);
    pr.print("cpu time", cpu_time);
    pr.print("free time", free_time);
    pr.print("total measured",transfer_time+free_time+kernel_time+result_time+cpu_time+free_time);
    return timestruct(0,transfer_time+result_time,cpu_time,free_time+malloc_time,kernel_time);
}


__global__ void startswith_index1_kernel(const gpuStringArray haystack, const char* needle, uint l, bool* result) {
    uint pos = blockDim.x*blockIdx.x + threadIdx.x;
    result[pos]=(haystack.first[pos]==needle[0]);
}

timestruct startswith_test::gpu_bench_with_index(const gpuStringArray& haystack, const char* needle, const int THREADS_PER_BLOCK) {
    assert(strlen(needle)<50);
    int c = 0;
    float kernel_time;
    float transfer_time;
    float result_time;
    float free_time;
    float total_time;
    float malloc_time;
    boost::timer::cpu_timer t2;
    {
        boost::timer::cpu_timer t;

        bool* result;
        bool* dev_result;
        char* dev_needle;
        gpuStringArray dev_strarr;
        dev_strarr.size = haystack.size;
        cudaEvent_t start, stop;

        uint noBlocks=haystack.size/THREADS_PER_BLOCK+1;
//        uint noChars=haystack.pos[haystack.size-1]+haystack.length[haystack.size-1];

        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord( start, 0 );
        CUDA_CHECK_RETURN(cudaMalloc((void**) &dev_result, sizeof(bool) * haystack.size));
    //    CUDA_CHECK_RETURN(cudaMalloc((void**) &dev_strarr.data, sizeof(char) * noChars));
    //    CUDA_CHECK_RETURN(cudaMalloc((void**) &dev_strarr.length, sizeof(uint) * haystack.size));
    //    CUDA_CHECK_RETURN(cudaMalloc((void**) &dev_strarr.pos, sizeof(uint) * haystack.size));
        CUDA_CHECK_RETURN(cudaMalloc((void**) &dev_strarr.first, sizeof(char) * haystack.size));
        CUDA_CHECK_RETURN(cudaMalloc((void**) &dev_needle, sizeof(char) * strlen(needle)));
#ifdef STRINGTEST_USE_PINNED_MEMORY
        CUDA_CHECK_RETURN(cudaHostAlloc(&result,haystack.size*sizeof(bool),cudaHostAllocDefault));
#else
        result = (bool*)malloc(haystack.size*sizeof(bool));
#endif
        cudaEventRecord( stop, 0 );

        cudaEventSynchronize( stop );
        cudaEventElapsedTime( &malloc_time, start, stop );
        cudaEventDestroy( start );
        cudaEventDestroy( stop );

        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord( start, 0 );
    //    CUDA_CHECK_RETURN(cudaMemcpy(dev_strarr.data, haystack.data, sizeof(char) * noChars, cudaMemcpyHostToDevice));
    //    CUDA_CHECK_RETURN(cudaMemcpy(dev_strarr.length, haystack.length, sizeof(uint) * haystack.size, cudaMemcpyHostToDevice));
    //    CUDA_CHECK_RETURN(cudaMemcpy(dev_strarr.pos, haystack.pos, sizeof(uint) * haystack.size, cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(cudaMemcpy(dev_strarr.first, haystack.first, sizeof(char) * haystack.size, cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(cudaMemcpy(dev_needle,needle,strlen(needle),cudaMemcpyHostToDevice));
        cudaEventRecord( stop, 0 );

        cudaEventSynchronize( stop );
        cudaEventElapsedTime( &transfer_time, start, stop );
        cudaEventDestroy( start );
        cudaEventDestroy( stop );


        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord( start, 0 );

        startswith_index1_kernel<<<noBlocks, THREADS_PER_BLOCK>>>(dev_strarr,dev_needle,strlen(needle),dev_result);
        cudaEventRecord( stop, 0 );

        cudaEventSynchronize( stop );
        cudaEventElapsedTime( &kernel_time, start, stop );
        cudaEventDestroy( start );
        cudaEventDestroy( stop );

        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord( start, 0 );
        CUDA_CHECK_RETURN(cudaMemcpy(result, dev_result, haystack.size*sizeof(bool), cudaMemcpyDeviceToHost));

        CUDA_CHECK_RETURN(cudaThreadSynchronize()); // Wait for the GPU launched work to complete
        CUDA_CHECK_RETURN(cudaGetLastError());
        cudaEventRecord( stop, 0 );

        cudaEventSynchronize( stop );
        cudaEventElapsedTime( &result_time, start, stop );
        cudaEventDestroy( start );
        cudaEventDestroy( stop );

        t2.start();
        uint l = strlen(needle);
#pragma omp parallel for reduction(+:c)
        for(int i = 0; i<haystack.size;++i) {
            if(result[i]) {
#ifdef USE_MEMCMP
                c+=(0==memcmp(&haystack.data[haystack.pos[i]],needle,l));
#else
                c+=strcmp2(&haystack.data[haystack.pos[i]],needle,l);
#endif
            }
        }
        t2.stop();


        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord( start, 0 );
        CUDA_CHECK_RETURN(cudaFree(dev_result));
//        CUDA_CHECK_RETURN(cudaFree(dev_strarr.data));
//        CUDA_CHECK_RETURN(cudaFree(dev_strarr.length));
        CUDA_CHECK_RETURN(cudaFree(dev_strarr.first));
//        CUDA_CHECK_RETURN(cudaFree(dev_strarr.pos));
        CUDA_CHECK_RETURN(cudaFree(dev_needle));
#ifdef STRINGTEST_USE_PINNED_MEMORY
        CUDA_CHECK_RETURN(cudaFreeHost(result));
#else
        free(result);
#endif
        cudaEventRecord( stop, 0 );

        cudaEventSynchronize( stop );
        cudaEventElapsedTime( &free_time, start, stop );
        cudaEventDestroy( start );
        cudaEventDestroy( stop );

        total_time = t.elapsed().wall;
    }

    total_time = total_time/1000000.0;
    float cpu_time = t2.elapsed().wall/1000000.0;

    _os<<"custom_string_arr on with index on GPU:"<<c<<" strings in haystack start with needle "<<needle<<std::endl;

    time_printer pr(total_time,_os);
    pr.print("alloc time", malloc_time);
    pr.print("transfer time",transfer_time);
    pr.print("kernel time",kernel_time);
    pr.print("result transfer",result_time);
    pr.print("cpu time", cpu_time);
    pr.print("free time", free_time);
    pr.print("total measured",transfer_time+free_time+kernel_time+result_time+cpu_time+free_time);
    return timestruct(cpu_time,transfer_time+result_time,0,free_time+malloc_time,kernel_time);
}

int callSimpleValueIDkernel( timestruct& t, const std::vector<uint>& valueIDs, int ent1, int ent2=0) {
    uint THREADS_PER_BLOCK = 256;

    int c;
	float kernel_time;
	float transfer_time;
	float result_time;
	float free_time;
	float malloc_time;
	boost::timer::cpu_timer t2;
	{
		boost::timer::cpu_timer t;

		bool* result;
		bool* dev_result;
		int* dev_vids;
		cudaEvent_t start, stop;

		uint noBlocks=valueIDs.size()/THREADS_PER_BLOCK+1;

		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord( start, 0 );
		CUDA_CHECK_RETURN(cudaMalloc((void**) &dev_result, sizeof(bool) * valueIDs.size()));
		CUDA_CHECK_RETURN(cudaMalloc((void**) &dev_vids, sizeof(int) * valueIDs.size()));
		CUDA_CHECK_RETURN(cudaHostAlloc(&result,valueIDs.size()*sizeof(bool),cudaHostAllocDefault));

#ifdef STRINGTEST_USE_PINNED_MEMORY
        CUDA_CHECK_RETURN(cudaHostAlloc(&result,_valueIDs.size()*sizeof(bool),cudaHostAllocDefault));
#else
        result = (bool*)malloc(valueIDs.size()*sizeof(bool));
#endif
		cudaEventRecord( stop, 0 );

		cudaEventSynchronize( stop );
		cudaEventElapsedTime( &malloc_time, start, stop );
		cudaEventDestroy( start );
		cudaEventDestroy( stop );

		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord( start, 0 );
		CUDA_CHECK_RETURN(cudaMemcpy(dev_vids, &valueIDs[0], sizeof(int) * valueIDs.size(), cudaMemcpyHostToDevice));
		cudaEventRecord( stop, 0 );

		cudaEventSynchronize( stop );
		cudaEventElapsedTime( &transfer_time, start, stop );
		cudaEventDestroy( start );
		cudaEventDestroy( stop );


		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord( start, 0 );

		if(0==ent2) {
			valueID_select_kernel<<<noBlocks, THREADS_PER_BLOCK>>>(dev_vids,valueIDs.size(),ent1,dev_result);
		} else {
			valueID_select_kernel<<<noBlocks, THREADS_PER_BLOCK>>>(dev_vids,valueIDs.size(),ent1,ent2,dev_result);
		}
		cudaEventRecord( stop, 0 );

		cudaEventSynchronize( stop );
		cudaEventElapsedTime( &kernel_time, start, stop );
		cudaEventDestroy( start );
		cudaEventDestroy( stop );

		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord( start, 0 );
		CUDA_CHECK_RETURN(cudaMemcpy(result, dev_result, valueIDs.size()*sizeof(bool), cudaMemcpyDeviceToHost));

		CUDA_CHECK_RETURN(cudaThreadSynchronize()); // Wait for the GPU launched work to complete
		CUDA_CHECK_RETURN(cudaGetLastError());
		cudaEventRecord( stop, 0 );

		cudaEventSynchronize( stop );
		cudaEventElapsedTime( &result_time, start, stop );
		cudaEventDestroy( start );
		cudaEventDestroy( stop );

		t2.start();
//#pragma omp parallel for reduction(+:c)
		for(int i = 0; i<valueIDs.size();++i) {
			c+=result[i];
		}
		t2.stop();


		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord( start, 0 );
		CUDA_CHECK_RETURN(cudaFree(dev_result));
		CUDA_CHECK_RETURN(cudaFree(dev_vids));
#ifdef STRINGTEST_USE_PINNED_MEMORY
		CUDA_CHECK_RETURN(cudaFreeHost(result));
#else
		free(result);
#endif
		cudaEventRecord( stop, 0 );

		cudaEventSynchronize( stop );
		cudaEventElapsedTime( &free_time, start, stop );
		cudaEventDestroy( start );
		cudaEventDestroy( stop );
	}

	t.kernel = kernel_time;
	t.overhead = (malloc_time + free_time);
//	std::cout<<malloc_time<<std::endl;
//	std::cout<<free_time<<std::endl;
//	exit(0);
	t.transfer = (result_time + transfer_time);
	t.seqtime = t2.elapsed().wall/1000000.0;
    return c;
}

timestruct startswith_test::gpu_bench(const dict_haystack& haystack, const char* needle, const int THREADS_PER_BLOCK) { /// @todo configurabel threadnum
    float res_time;
    float dict_time = 0.0;
    float value_time = 0.0;
    int c = 0;
    timestruct tstr;
    {
        boost::timer::auto_cpu_timer t(_os);
        boost::timer::cpu_timer t_part;
        uint l = strlen(needle);
        int ent1 = haystack._dict.lower_bound(0,haystack._dict.size,needle,l);

        if(ent1<haystack._dict.size && gpuStringArray::strcmp2_cpu(haystack._dict[ent1],needle,l)) {
            char* needle2 = (char*)malloc((strlen(needle)+1));
            strcpy(needle2,needle);
            needle2[strlen(needle)-1]++;
            int ent2 = haystack._dict.upper_bound(ent1,haystack._dict.size,needle2,l);
            dict_time = t_part.elapsed().wall;
            if(ent2==ent1+1) {
            	c = callSimpleValueIDkernel(tstr,haystack._valueIDs,ent1);
            } else {
            	c = callSimpleValueIDkernel(tstr,haystack._valueIDs,ent1,ent2);
            }
        } else {
            dict_time = t_part.elapsed().wall;
        }
        res_time=t.elapsed().wall;
    }
    _os<<"dict on parallel CPU: "<<c<<" strings in haystack start with needle "<<needle<<std::endl;

    time_printer pr(res_time/1000000.0,_os);
    pr.print("dict time",dict_time/1000000.0);
    pr.print("valueID time",value_time/1000000.0);
    tstr.seqtime += dict_time/1000000.0;
    return tstr;
}
