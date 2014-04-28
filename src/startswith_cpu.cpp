#include "emmintrin.h"
#include "stringtest.hpp"


timestruct startswith_test::cpu_bench(std::vector<std::string>& haystack, const char* needle) {
    float res_time;
    int c = 0;
    {
        boost::timer::auto_cpu_timer t(_os);
        std::string n(needle);
        for(int i = 0; i<haystack.size();++i) {
            c+=(0==haystack[i].compare(0,n.length(),n));
        }
        res_time = t.elapsed().wall;
    }
    _os<<"std_string on CPU: "<<c<<" strings in haystack start with needle "<<needle<<std::endl;
    return timestruct(0,0,res_time/1000000.0);
}

timestruct startswith_test::cpu_bench_parallel(std::vector<std::string>& haystack, const char* needle) {
    float res_time;
    int c = 0;
    {
        boost::timer::auto_cpu_timer t(_os);
        std::string n(needle);
#pragma omp parallel for reduction(+:c)
        for(int i = 0; i<haystack.size();++i) {
            c+=(0==haystack[i].compare(0,n.length(),n));
        }
        res_time = t.elapsed().wall;
    }
    _os<<"std_string parallel on CPU: "<<c<<" strings in haystack start with needle "<<needle<<std::endl;
    return timestruct(res_time/1000000.0);
}

timestruct startswith_test::cpu_bench(const gpuStringArray& haystack, const char* needle) {
    float res_time;
    int c = 0;
    {
        boost::timer::auto_cpu_timer t(_os);
        uint l = strlen(needle);
        for(int i = 0; i<haystack.size;++i) {
            if(haystack.length[i]>=l) {
#ifdef USE_MEMCMP
                c+=(0==memcmp(&haystack.data[haystack.pos[i]],needle,l));
#else
                c+=gpuStringArray::strcmp2_cpu(&haystack.data[haystack.pos[i]],needle,l);
#endif
            }
        }
        res_time = t.elapsed().wall;
    }
    _os<<"custom_string_arr on CPU: "<<c<<" strings in haystack start with needle "<<needle<<std::endl;
    return timestruct(0,0,res_time/1000000.0);
}

timestruct startswith_test::cpu_bench_parallel(const gpuStringArray& haystack, const char* needle) {
    float res_time;
    int c = 0;
    {
        boost::timer::auto_cpu_timer t(_os);
        uint l = strlen(needle);
#pragma omp parallel for reduction(+:c)
        for(int i = 0; i<haystack.size;++i) {
            if(haystack.length[i]>=l) {
                c+=gpuStringArray::strcmp2_cpu(&haystack.data[haystack.pos[i]],needle,l);
            }
        }
        res_time = t.elapsed().wall;
    }
    _os<<"custom_string_arr parallel on CPU: "<<c<<" strings in haystack start with needle "<<needle<<std::endl;
    return timestruct(res_time/1000000.0);
}

timestruct startswith_test::cpu_bench_with_index(const gpuStringArray& haystack, const char* needle) {
    float res_time;
    int c = 0;
    {
        boost::timer::auto_cpu_timer t(_os);
        uint l = strlen(needle);
        for(int i = 0; i<haystack.size;++i) {
            if(haystack.length[i]>=l && haystack.first[i]==needle[0]) {
#ifdef USE_MEMCMP
                c+=(0==memcmp(&haystack.data[haystack.pos[i]],needle,l));
#else
                c+=gpuStringArray::strcmp2_cpu(&haystack.data[haystack.pos[i]],needle,l);
#endif
            }
        }
        res_time = t.elapsed().wall;
    }
    _os<<"custom_string_arr on CPU with index: "<<c<<" strings in haystack start with needle "<<needle<<std::endl;
    return timestruct(0,0,res_time/1000000.0);
}

timestruct startswith_test::cpu_bench_with_index_SSE(const gpuStringArray& haystack, const char* needle) {
    float res_time;
    char needle16c[16];
    for(int i=0;i<16;++i) {
        needle16c[i]=needle[0];
    }
    __m128i  needle16 = _mm_loadu_si128( reinterpret_cast<__m128i*>( &needle16c ) );
    int c = 0;
    {
        boost::timer::auto_cpu_timer t(_os);
        uint l = strlen(needle);
        for(int i = 0; i<haystack.size-16;i+=16) {
            __m128i firsts = _mm_loadu_si128( reinterpret_cast<__m128i*>( haystack.first+i ) );
            uint sse_res = _mm_movemask_epi8( _mm_cmpeq_epi8( firsts, needle16 ) );;
            if(sse_res!=0) {
                for(int j = i; j<i+16;++j) {
                    if(haystack.length[j]>=l && haystack.first[j]==needle[0]) {
        #ifdef USE_MEMCMP
                        c+=(0==memcmp(&haystack.data[haystack.pos[j]],needle,l));
        #else
                        c+=gpuStringArray::strcmp2_cpu(&haystack.data[haystack.pos[j]],needle,l);
        #endif
                    }
                }
            }
        }

        //TODO: don't forget the last <16 elements
        res_time = t.elapsed().wall;
    }
    _os<<"custom_string_arr on CPU with index and SSE: "<<c<<" strings in haystack start with needle "<<needle<<std::endl;
    return timestruct(0,0,res_time/1000000.0);
}



timestruct startswith_test::cpu_bench_parallel_with_index(const gpuStringArray& haystack, const char* needle) {
    float res_time;
    int c = 0;
    {
        boost::timer::auto_cpu_timer t(_os);
        uint l = strlen(needle);
#pragma omp parallel for reduction(+:c)
        for(int i = 0; i<haystack.size;++i) {
            if(haystack.length[i]>=l && haystack.first[i]==needle[0]) {
#ifdef USE_MEMCMP
                c+=(0==memcmp(&haystack.data[haystack.pos[i]],needle,l));
#else
                c+=gpuStringArray::strcmp2_cpu(&haystack.data[haystack.pos[i]],needle,l);
#endif
            }
        }
        res_time = t.elapsed().wall;
    }
    _os<<"custom_string_arr on parallel CPU with index: "<<c<<" strings in haystack start with needle "<<needle<<std::endl;
    return timestruct(res_time/1000000.0);
}


timestruct startswith_test::cpu_bench_parallel_with_index_SSE(const gpuStringArray& haystack, const char* needle) {
    float res_time;
    char needle16c[16];
    for(int i=0;i<16;++i) {
        needle16c[i]=needle[0];
    }
    __m128i  needle16 = _mm_loadu_si128( reinterpret_cast<__m128i*>( &needle16c ) );
    int c = 0;
    {
        boost::timer::auto_cpu_timer t(_os);
        uint l = strlen(needle);
#pragma omp parallel for reduction(+:c)
        for(int i = 0; i<haystack.size-16;i+=16) {
            __m128i firsts = _mm_loadu_si128( reinterpret_cast<__m128i*>( haystack.first+i ) );
            uint sse_res = _mm_movemask_epi8( _mm_cmpeq_epi8( firsts, needle16 ) );;
            if(sse_res!=0) {
                for(int j = i; j<i+16;++j) {
                    if(haystack.length[j]>=l && haystack.first[j]==needle[0]) {
        #ifdef USE_MEMCMP
                        c+=(0==memcmp(&haystack.data[haystack.pos[j]],needle,l));
        #else
                        c+=gpuStringArray::strcmp2_cpu(&haystack.data[haystack.pos[j]],needle,l);
        #endif
                    }
                }
            }
        }

        //TODO: don't forget the last <16 elements
        res_time = t.elapsed().wall;
    }
    _os<<"custom_string_arr on parallel CPU with index and SSE: "<<c<<" strings in haystack start with needle "<<needle<<std::endl;
    return timestruct(res_time/1000000.0);
}

timestruct startswith_test::cpu_bench(const dict_haystack& haystack, const char* needle) {
    float res_time;
    float dict_time = 0.0;
    float value_time = 0.0;
    int c = 0;
    {
        boost::timer::auto_cpu_timer t(_os);
        boost::timer::cpu_timer t_part;
        uint l = strlen(needle);
        int ent1 = haystack._dict.lower_bound(0,haystack._dict.size,needle,l);
//        std::cout<<ent1<<" "<<haystack.dict.size<<std::endl;
        if(ent1<haystack._dict.size && gpuStringArray::strcmp2_cpu(haystack._dict[ent1],needle,l)) {
            char* needle2 = (char*)malloc((strlen(needle)+1));
            strcpy(needle2,needle);
            needle2[strlen(needle)-1]++;
            int ent2 = haystack._dict.upper_bound(ent1,haystack._dict.size,needle2,l);
            dict_time = t_part.elapsed().wall;
            t_part.start();
            if(ent2==ent1+1) {
                for(int i = 0; i<haystack._valueIDs.size();++i) {
                    c+=(haystack._valueIDs[i]==ent1);
                }
            } else {
                for(int i = 0; i<haystack._valueIDs.size();++i) {
                    c+=(haystack._valueIDs[i]<ent2 && haystack._valueIDs[i]>=ent1);
                }
            }
            value_time = t_part.elapsed().wall;
        } else {
            dict_time = t_part.elapsed().wall;
        }
        res_time=t.elapsed().wall;
    }
    _os<<"dict on parallel CPU: "<<c<<" strings in haystack start with needle "<<needle<<std::endl;

    time_printer pr(res_time/1000000.0,_os);
    pr.print("dict time",dict_time/1000000.0);
    pr.print("valueID time",value_time/1000000.0);
    return timestruct(0,0,value_time/1000000.0+dict_time/1000000.0);
}

timestruct startswith_test::cpu_bench_parallel(const dict_haystack& haystack, const char* needle) {
    float res_time;
    float dict_time = 0.0;
    float value_time = 0.0;
    int c = 0;
    {
        boost::timer::auto_cpu_timer t(_os);
        boost::timer::cpu_timer t_part;
        uint l = strlen(needle);
        int ent1 = haystack._dict.lower_bound(0,haystack._dict.size,needle,l);
//        std::cout<<ent1<<" "<<haystack.dict.size<<std::endl;
        if(ent1<haystack._dict.size && gpuStringArray::strcmp2_cpu(haystack._dict[ent1],needle,l)) {
            char* needle2 = (char*)malloc((strlen(needle)+1));
            strcpy(needle2,needle);
            needle2[strlen(needle)-1]++;
            int ent2 = haystack._dict.upper_bound(ent1,haystack._dict.size,needle2,l);
            dict_time = t_part.elapsed().wall;
            t_part.start();
            if(ent2==ent1+1) {
#pragma omp parallel for reduction(+:c)
                for(int i = 0; i<haystack._valueIDs.size();++i) {
                    c+=(haystack._valueIDs[i]==ent1);
                }
            } else {
#pragma omp parallel for reduction(+:c)
                for(int i = 0; i<haystack._valueIDs.size();++i) {
                    c+=(haystack._valueIDs[i]<ent2 && haystack._valueIDs[i]>=ent1);
                }
            }
            value_time = t_part.elapsed().wall;
        } else {
            dict_time = t_part.elapsed().wall;
        }
        res_time=t.elapsed().wall;
    }
    _os<<"dict on parallel CPU: "<<c<<" strings in haystack start with needle "<<needle<<std::endl;

    time_printer pr(res_time/1000000.0,_os);
    pr.print("dict time",dict_time/1000000.0);
    pr.print("valueID time",value_time/1000000.0);
    return timestruct(value_time/1000000.0,0,dict_time/1000000.0);
}
