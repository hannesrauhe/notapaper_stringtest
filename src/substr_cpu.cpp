/**
 * author: Hannes Rauhe
 * Using the Horspool boyer moore algorithm from https://github.com/FooBarWidget/boyer-moore-horspool
 */

#include "stringtest.hpp"
#include "strstr/Horspool.hpp"
#include <algorithm>

timestruct substr_test::cpu_bench(std::vector<std::string>& haystack, const char* needle) {
    float res_time;
    int c = 0;
    {
        boost::timer::auto_cpu_timer t(_os);
//        std::string n(needle);
        uint l1 = strlen(needle);
        occtable_type occ = CreateOccTable((const unsigned char*)needle, l1);
        for(int i = 0; i<haystack.size();++i) {
//            c+=(std::string::npos!=haystack[i].find(n));
            uint l2 = haystack[i].size();
            c+=(l2!=SearchInHorspool((const unsigned char*)haystack[i].c_str(),l2,occ,(const unsigned char*)needle,l1));
        }
        res_time = t.elapsed().wall;
    }
    _os<<"std_string on CPU: "<<c<<" strings in haystack start with needle "<<needle<<std::endl;
    return timestruct(0,0,res_time/1000000.0);
}

timestruct substr_test::cpu_bench_parallel(std::vector<std::string>& haystack, const char* needle) {
    float res_time;
    int c = 0;
    {
        boost::timer::auto_cpu_timer t(_os);
//        std::string n(needle);
        uint l1 = strlen(needle);
        occtable_type occ = CreateOccTable((const unsigned char*)needle, l1);
#pragma omp parallel for reduction(+:c)
        for(int i = 0; i<haystack.size();++i) {
//            c+=(std::string::np(_os!=haystack[i].find(n));
            uint l2 = haystack[i].size();
            c+=(l2!=SearchInHorspool((const unsigned char*)haystack[i].c_str(),l2,occ,(const unsigned char*)needle,l1));
        }
        res_time = t.elapsed().wall;
    }
    _os<<"std_string parallel on CPU: "<<c<<" strings in haystack contain the needle "<<needle<<std::endl;
    return res_time/1000000.0;
}

timestruct substr_test::cpu_bench(const gpuStringArray& haystack, const char* needle) {
    float res_time;
    int c = 0;
    {
        boost::timer::auto_cpu_timer t(_os);
        uint l1 = strlen(needle);
        occtable_type occ = CreateOccTable((const unsigned char*)needle, l1);
        for(int i = 0; i<haystack.size;++i) {
            uint l2 = haystack.length[i];
            if(l2>=l1) {
//                c+=(true==boyer_moore(&haystack.data[haystack.pos[i]],l2,needle,l1));
//                c+=(NULL!=boyermoore_horspool_memmem(&haystack.data[haystack.pos[i]],l2,needle,l1));
//                c+=(NULL!=memmem(&haystack.data[haystack.pos[i]],l2,needle,l1));
                c+=(l2!=SearchInHorspool((const unsigned char*)&haystack.data[haystack.pos[i]],l2,occ,(const unsigned char*)needle,l1));
            }
        }
        res_time = t.elapsed().wall;
    }
    _os<<"custom_string_arr on CPU: "<<c<<" strings in haystack start with needle "<<needle<<std::endl;
    return timestruct(0,0,res_time/1000000.0);
}

timestruct substr_test::cpu_bench_parallel(const gpuStringArray& haystack, const char* needle) {
    float res_time;
    int c = 0;
    {
        boost::timer::auto_cpu_timer t(_os);
        uint l1 = strlen(needle);
        occtable_type occ = CreateOccTable((const unsigned char*)needle, l1);
#pragma omp parallel for reduction(+:c)
        for(int i = 0; i<haystack.size;++i) {
            uint l2 = haystack.length[i];
            if(l2>=l1) {
//                c+=(true==boyer_moore(&haystack.data[haystack.pos[i]],l2,needle,l1));
//                c+=(NULL!=boyermoore_horspool_memmem(&haystack.data[haystack.pos[i]],l2,needle,l1));
//                c+=(NULL!=memmem(&haystack.data[haystack.pos[i]],l2,needle,l1));
                c+=(l2!=SearchInHorspool((const unsigned char*)&haystack.data[haystack.pos[i]],l2,occ,(const unsigned char*)needle,l1));
            }
        }
        res_time = t.elapsed().wall;
    }
    _os<<"custom_string_arr parallel on CPU: "<<c<<" strings in haystack contain the needle "<<needle<<std::endl;
    return res_time/1000000.0;
}

timestruct substr_test::cpu_bench(const dict_haystack& haystack, const char* needle) {
    float res_time;
    float dict_time = 0.0;
    float value_time = 0.0;
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

		for(int i = 0; i<haystack._valueIDs.size(); ++i) {
			c+=(true==matching[haystack._valueIDs[i]]);
		}
		free(matching);
        value_time = t_part.elapsed().wall;
        res_time=t.elapsed().wall;
    }
    _os<<"dict on CPU (sequential): "<<c<<" strings in haystack contain the needle "<<needle<<std::endl;

    time_printer pr(res_time/1000000.0,_os);
    pr.print("dict time",dict_time/1000000.0);
    pr.print("valueID time",value_time/1000000.0);

    return timestruct(0,0,res_time/1000000.0);
}

timestruct substr_test::cpu_bench_parallel(const dict_haystack& haystack, const char* needle) {
    float res_time;
    float dict_time = 0.0;
    float value_time = 0.0;
    int c = 0;
    {
        boost::timer::auto_cpu_timer t(_os);
        boost::timer::cpu_timer t_part;
        uint l1 = strlen(needle);
        bool* matching = (bool*)malloc(haystack._dict.size);
        occtable_type occ = CreateOccTable((const unsigned char*)needle, l1);

#pragma omp parallel for
        for(int i = 0; i<haystack._dict.size;++i) {
            uint l2 = haystack._dict.length[i];
			matching[i]= l2>=l1 && l2!=SearchInHorspool((const unsigned char*)&haystack._dict.data[haystack._dict.pos[i]],l2,occ,(const unsigned char*)needle,l1);
        }
        dict_time = t_part.elapsed().wall;
        t_part.start();

#pragma omp parallel for
		for(int i = 0; i<haystack._valueIDs.size(); ++i) {
			c+=(true==matching[haystack._valueIDs[i]]);
		}
		free(matching);
        value_time = t_part.elapsed().wall;
        res_time=t.elapsed().wall;
    }
    _os<<"dict on parallel CPU: "<<c<<" strings in haystack contain the needle "<<needle<<std::endl;

    time_printer pr(res_time/1000000.0,_os);
    pr.print("dict time",dict_time/1000000.0);
    pr.print("valueID time",value_time/1000000.0);

    return res_time/1000000.0;
}
