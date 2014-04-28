#include "stringtest.hpp"
#include "strstr/Horspool.hpp"
#include "substr_gpu_kernels.cuh"
#include <algorithm>

timestruct substr_test::gpu_bench(const gpuStringArray& haystack, const char* needle, const int THREADS_PER_BLOCK) {
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


        uint* result;
        uint* dev_result;
        size_t* dev_occ;
        uint noBlocks=haystack.size/THREADS_PER_BLOCK+1;
        uint noChars=haystack.pos[haystack.size-1]+haystack.length[haystack.size-1];
        int smemSize = (THREADS_PER_BLOCK <= 32) ? 2 * THREADS_PER_BLOCK * sizeof(uint) : THREADS_PER_BLOCK * sizeof(uint);

        occtable_type occ = CreateOccTable((const unsigned char*)needle, strlen(needle));

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
        CUDA_CHECK_RETURN(cudaMalloc((void**) &dev_occ, sizeof(size_t) * occ.size()));
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
        CUDA_CHECK_RETURN(cudaMemcpy(dev_occ,&occ[0],sizeof(size_t) * occ.size(),cudaMemcpyHostToDevice));
        t2.stop();
        transfer_time += t2.elapsed().wall/1000000.0;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord( start, 0 );
        switch (THREADS_PER_BLOCK)
        {
            case 512:
                substr_kernel_opt<512, false><<< noBlocks, THREADS_PER_BLOCK, smemSize >>>(dev_strarr,dev_needle,strlen(needle),dev_occ,haystack.size,dev_result); break;
            case 256:
                substr_kernel_opt<256, false><<< noBlocks, THREADS_PER_BLOCK, smemSize >>>(dev_strarr,dev_needle,strlen(needle),dev_occ,haystack.size,dev_result); break;
            case 128:
                substr_kernel_opt<128, false><<< noBlocks, THREADS_PER_BLOCK, smemSize >>>(dev_strarr,dev_needle,strlen(needle),dev_occ,haystack.size,dev_result); break;
            case 64:
                substr_kernel_opt< 64, false><<< noBlocks, THREADS_PER_BLOCK, smemSize >>>(dev_strarr,dev_needle,strlen(needle),dev_occ,haystack.size,dev_result); break;
            case 32:
                substr_kernel_opt< 32, false><<< noBlocks, THREADS_PER_BLOCK, smemSize >>>(dev_strarr,dev_needle,strlen(needle),dev_occ,haystack.size,dev_result); break;
            case 16:
                substr_kernel_opt< 16, false><<< noBlocks, THREADS_PER_BLOCK, smemSize >>>(dev_strarr,dev_needle,strlen(needle),dev_occ,haystack.size,dev_result); break;
            case  8:
                substr_kernel_opt<  8, false><<< noBlocks, THREADS_PER_BLOCK, smemSize >>>(dev_strarr,dev_needle,strlen(needle),dev_occ,haystack.size,dev_result); break;
            case  4:
                substr_kernel_opt<  4, false><<< noBlocks, THREADS_PER_BLOCK, smemSize >>>(dev_strarr,dev_needle,strlen(needle),dev_occ,haystack.size,dev_result); break;
            case  2:
                substr_kernel_opt<  2, false><<< noBlocks, THREADS_PER_BLOCK, smemSize >>>(dev_strarr,dev_needle,strlen(needle),dev_occ,haystack.size,dev_result); break;
            case  1:
                substr_kernel_opt<  1, false><<< noBlocks, THREADS_PER_BLOCK, smemSize >>>(dev_strarr,dev_needle,strlen(needle),dev_occ,haystack.size,dev_result); break;
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
        CUDA_CHECK_RETURN(cudaFree(dev_occ));
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

    _os<<"custom_string_arr on GPU:      "<<c<<" strings in haystack contain the needle "<<needle<<std::endl;

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

/*
timestruct substr_test::gpu_bench(const dict_haystack& haystack, const char* needle, const int THREADS_PER_BLOCK) {
    float res_time = 0.0;
    float dict_time = 0.0;
    float transfer_time = 0.0;
    float result_time = 0.0;
    float cpu_time = 0.0;
    float free_time = 0.0;
    int c = 0;
    {
        boost::timer::auto_cpu_timer t(_os);
        boost::timer::cpu_timer t_part;
        uint l1 = strlen(needle);

        bool* matching = (bool*)malloc(haystack._dict.size);
        occtable_type occ = CreateOccTable((const unsigned char*)needle, l1);

        for(int i = 0; i<haystack._dict.size;++i) {
            uint l2 = haystack._dict.length[i];
			matching[i]= l2>=l1 && l2!=SearchInHorspool((const unsigned char*)&haystack._dict.data[haystack._dict.pos[i]],l2,occ,(const unsigned char*)needle,l1);
        }
        dict_time = t_part.elapsed().wall;
        t_part.start();
//        std::vector<uint> matching_vIDs;
//        occtable_type occ = CreateOccTable((const unsigned char*)needle, l1);
//        for(int i = 0; i<haystack._dict.size;++i) {
//            uint l2 = haystack._dict.length[i];
//            if(l2>=l1 && l2!=SearchInHorspool((const unsigned char*)&haystack._dict.data[haystack._dict.pos[i]],l2,occ,(const unsigned char*)needle,l1)) {
//                matching_vIDs.push_back(i);
//            }
//        }
//        dict_time = t_part.elapsed().wall;
//        t_part.start();
//
//        if(!matching_vIDs.empty()) {
//            if(matching_vIDs.size()>1) {
//                std::sort(matching_vIDs.begin(),matching_vIDs.end());
//            }
//            bool* result;
//            bool* dev_result;
//            int* dev_mvIDs;
//            int* dev_vIDs;
//            cudaEvent_t start, stop;
//
//            uint noBlocks=haystack._valueIDs.size()/THREADS_PER_BLOCK+1;
//
//            cudaEventCreate(&start);
//            cudaEventCreate(&stop);
//            cudaEventRecord( start, 0 );
//            CUDA_CHECK_RETURN(cudaMalloc((void**) &dev_result, sizeof(bool) * haystack._valueIDs.size()));
//            CUDA_CHECK_RETURN(cudaMalloc((void**) &dev_vIDs, sizeof(int) * haystack._valueIDs.size()));
//            if(matching_vIDs.size()>1) {
//                CUDA_CHECK_RETURN(cudaMalloc((void**) &dev_mvIDs, sizeof(int) * matching_vIDs.size()));
//                CUDA_CHECK_RETURN(cudaMemcpy(dev_mvIDs,&matching_vIDs[0], sizeof(int) * matching_vIDs.size(),cudaMemcpyHostToDevice));
//            }
//            CUDA_CHECK_RETURN(cudaMemcpy(dev_vIDs,&haystack._valueIDs[0], sizeof(int) * haystack._valueIDs.size(),cudaMemcpyHostToDevice));
//            cudaEventRecord( stop, 0 );
//
//            cudaEventSynchronize( stop );
//            cudaEventElapsedTime( &transfer_time, start, stop );
//            cudaEventDestroy( start );
//            cudaEventDestroy( stop );
//
//            if(matching_vIDs.size()>1) {
//                CUDA_CHECK_RETURN(cudaMalloc((void**) &dev_mvIDs, sizeof(int) * matching_vIDs.size()));
//                CUDA_CHECK_RETURN(cudaMemcpy(dev_mvIDs,&matching_vIDs[0], sizeof(int) * matching_vIDs.size(),cudaMemcpyHostToDevice));
//    //
//    //
//    //
//            } else if(matching_vIDs.size()==1)  {
//                int v = matching_vIDs[0];
//    //
//    //
//    //
//            }
//            cudaEventCreate(&start);
//            cudaEventCreate(&stop);
//            cudaEventRecord( start, 0 );
//            CUDA_CHECK_RETURN(cudaHostAlloc(&result,haystack._valueIDs.size()*sizeof(bool),cudaHostAllocDefault));
//            CUDA_CHECK_RETURN(cudaMemcpy(result, dev_result, haystack._valueIDs.size()*sizeof(bool), cudaMemcpyDeviceToHost));
//
//            CUDA_CHECK_RETURN(cudaThreadSynchronize()); // Wait for the GPU launched work to complete
//            CUDA_CHECK_RETURN(cudaGetLastError());
//            cudaEventRecord( stop, 0 );
//
//            cudaEventSynchronize( stop );
//            cudaEventElapsedTime( &result_time, start, stop );
//            cudaEventDestroy( start );
//            cudaEventDestroy( stop );
//
//            t_part.start();
//            for (uint i = 0; i < haystack._valueIDs.size(); i++)
//               c+=(true==result[i]);
//            cpu_time = t_part.elapsed().wall;
//
//
//            cudaEventCreate(&start);
//            cudaEventCreate(&stop);
//            cudaEventRecord( start, 0 );
//            CUDA_CHECK_RETURN(cudaFree(dev_result));
//            CUDA_CHECK_RETURN(cudaFree(dev_vIDs));
//            if(matching_vIDs.size()>1) {
//                CUDA_CHECK_RETURN(cudaFree(dev_mvIDs));
//            }
//            cudaEventRecord( stop, 0 );
//
//            cudaEventSynchronize( stop );
//            cudaEventElapsedTime( &free_time, start, stop );
//            cudaEventDestroy( start );
//            cudaEventDestroy( stop );
        }
        res_time=t.elapsed().wall;
    }
    _os<<"dict on parallel GPU: "<<c<<" strings in haystack contain needle "<<needle<<std::endl;

    time_printer pr(res_time,_os);
    pr.print("dict time",dict_time/1000000.0);
    pr.print("transfer time",transfer_time);
//    pr.print("kernel time",kernel_time);
    pr.print("result transfer",result_time);
    pr.print("cpu time", cpu_time);
    pr.print("free time", free_time);

    return res_time/1000000.0;
}

*/
