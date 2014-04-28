#include "stringtest.hpp"

void gpureset() {
    cudaDeviceReset();
}

void gpuStringArray::init(std::vector<std::string>& haystack, int noChar) {
//    boost::timer::auto_cpu_timer t;
    size = haystack.size();
    byte_size = noChar + haystack.size()*sizeof(uint)*2;
    CUDA_CHECK_RETURN(cudaHostAlloc(&length,haystack.size()*sizeof(uint),cudaHostAllocDefault));
    CUDA_CHECK_RETURN(cudaHostAlloc(&pos,haystack.size()*sizeof(uint),cudaHostAllocDefault));
    //space for last terminating character because it's easier
    CUDA_CHECK_RETURN(cudaHostAlloc(&data,(1+noChar)*sizeof(char),cudaHostAllocDefault));
    CUDA_CHECK_RETURN(cudaHostAlloc(&first,haystack.size()*sizeof(char),cudaHostAllocDefault));

    int cpos = 0;
    for(int i = 0;i<size;++i) {
        pos[i]=cpos;
        length[i]=haystack[i].length();
        first[i]=haystack[i][0];
        strcpy(data+cpos, haystack[i].c_str());
        cpos+=haystack[i].length();
    }
}

void gpuStringArray::destroy() {
    byte_size = 0;
    CUDA_CHECK_RETURN(cudaFreeHost(data));
    CUDA_CHECK_RETURN(cudaFreeHost(pos));
    CUDA_CHECK_RETURN(cudaFreeHost(length));
    CUDA_CHECK_RETURN(cudaFreeHost(first));
}
