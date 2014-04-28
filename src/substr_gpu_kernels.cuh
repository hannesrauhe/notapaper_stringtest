/***
 * @author Hannes Rauhe
 *
 * @brief mostly taken from the nvidia reduction sample
 */



__host__ __device__ bool strcmp3(const char* str1, const char* str2, int length) {
    for(int i = 0; i<length; ++i) {
        if(str1[i]!=str2[i]) {
            return false;
        }
    }
    return true;
}

/***
 * horspool copied/modfied from the CPU part originally written by Joel Yliluoma <joel.yliluoma@iki.fi> - see horspool.cpp in strstr subdir
 *
 */
__device__ bool horspool(const char* str1, int l1, const char* needle, int needle_length, const size_t* dev_occ) {
    if(needle_length > l1) return false;
    if(needle_length == 1)
    {
        return false; ///@todo special case
    }

    const size_t needle_length_minus_1 = needle_length-1;

    const unsigned char last_needle_char = needle[needle_length_minus_1];

    size_t haystack_position=0;
    while(haystack_position <= l1-needle_length)
    {
        const unsigned char occ_char = str1[haystack_position + needle_length_minus_1];

        if(last_needle_char == occ_char
        && strcmp3(needle, str1+haystack_position, needle_length_minus_1))
        {
            return true;
        }
        haystack_position += dev_occ[occ_char];
    }
    return false;
}


/*
    The following kernels are modified from the CUDA samples
*/
// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory
{
    __device__ inline operator       T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};

// specialize for double to avoid unaligned memory
// access compile errors
template<>
struct SharedMemory<double>
{
    __device__ inline operator       double *()
    {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }

    __device__ inline operator const double *() const
    {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }
};

template <unsigned int blockSize, bool nIsPow2>
__global__ void
substr_kernel_opt(const gpuStringArray haystack, const char* needle, uint l, const size_t* dev_occ, uint n, uint* result)
{
    uint *sdata = SharedMemory<uint>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;

    uint mySum = 0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
        mySum += horspool(&haystack.data[haystack.pos[i]],haystack.length[i],needle,l,dev_occ);

        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n) {
            mySum += horspool(&haystack.data[haystack.pos[i+blockSize]],haystack.length[i+blockSize],needle,l,dev_occ);
        }

        i += gridSize;
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = mySum;
    __syncthreads();


    // do reduction in shared mem
    if (blockSize >= 512)
    {
        if (tid < 256)
        {
            sdata[tid] = mySum = mySum + sdata[tid + 256];
        }

        __syncthreads();
    }

    if (blockSize >= 256)
    {
        if (tid < 128)
        {
            sdata[tid] = mySum = mySum + sdata[tid + 128];
        }

        __syncthreads();
    }

    if (blockSize >= 128)
    {
        if (tid <  64)
        {
            sdata[tid] = mySum = mySum + sdata[tid +  64];
        }

        __syncthreads();
    }

    if (tid < 32)
    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile uint *smem = sdata;

        if (blockSize >=  64)
        {
            smem[tid] = mySum = mySum + smem[tid + 32];
        }

        if (blockSize >=  32)
        {
            smem[tid] = mySum = mySum + smem[tid + 16];
        }

        if (blockSize >=  16)
        {
            smem[tid] = mySum = mySum + smem[tid +  8];
        }

        if (blockSize >=   8)
        {
            smem[tid] = mySum = mySum + smem[tid +  4];
        }

        if (blockSize >=   4)
        {
            smem[tid] = mySum = mySum + smem[tid +  2];
        }

        if (blockSize >=   2)
        {
            smem[tid] = mySum = mySum + smem[tid +  1];
        }
    }

    // write result for this block to global mem
    if (tid == 0)
        result[blockIdx.x] = sdata[0];
}
