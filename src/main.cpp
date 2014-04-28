#include "stringtest.hpp"
#include <sstream>
#include <boost/program_options.hpp>

#define CSV_OUTPUT


void read_haystack_varlen(const char* filename, std::vector<std::string>& haystack, int& minlen, int& maxlen, int& noChar) {
    std::string buffer;
    int size = 0;
    std::ifstream value_file(filename);
    if(!value_file.good()) {
        std::cerr<<"file "<<filename<<" does not exist"<<std::endl;
        return;
    }

    while(getline(value_file, buffer)) {
        ++size;
    }

    value_file.close();

    haystack.resize(size);
    value_file.open(filename);
    minlen = std::numeric_limits<int>::max();
    maxlen = 0;
    noChar = 0;
    for(int i = 0; i<size; ++i) {
        std::getline(value_file,haystack[i]);
        if(haystack[i].size()>maxlen) maxlen=haystack[i].size();
        if(haystack[i].size()<minlen) minlen=haystack[i].size();
        noChar+=haystack[i].size();
    }
}

void test_lower_bound() {
    const char* needle = "furiously expre";
    std::vector<std::string> testhay;
    testhay.push_back("furiously expr");
    testhay.push_back("furiously expre");
    testhay.push_back("furiously expres");
    testhay.push_back("furiously express");
    testhay.push_back("furiously express ");
    testhay.push_back("furiously express T");
    testhay.push_back("furiously express Ti");
    testhay.push_back("furiously express a");
    testhay.push_back("furiously express ac");
    testhay.push_back("furiously express acc");

    gpuStringArray gpu_testhay;
    gpu_testhay.init(testhay, 1000);

    std::cout<<"is: "<<gpu_testhay[gpu_testhay.lower_bound(0,10,needle,strlen(needle))]<<std::endl;
    std::cout<<"should be: "<<*std::lower_bound(testhay.begin(),testhay.end(),needle);
    exit(0);
}

void omp_bug_workaround() {
    //omp parallel gets stuck after building the dict <-- workaround
    boost::timer::cpu_timer t_work;
    int workaround_i = 0;
    #pragma omp parallel for
    for(int i = 0;i<6000000;++i) {
//            if(!gpuStringArray::strcmp2_cpu(haystack[i].c_str(),cpu_d_haystack[i],haystack[i].size())) {
//                std::cerr<<"something went wrong when creating the dict: pos: "<<i<<haystack[i]<<std::endl;
//            }
        workaround_i+=i;
    }
    std::cout<<workaround_i<<" <-- This workaround took"<<t_work.format()<<std::endl;
}

int main(int ac, char** av) {
	namespace po = boost::program_options;

    std::string filename = "/home/hannes/tpch/lineitem/L_COMMENT";
    std::string needle = "the";//"furiously expre";
    std::string testname = "substr";
    int use_gpu = 1;
    int test_iterations = 100;
    int verbose = 0;
    int threadnum = 256;
    bool measure_transfer = false;

    // Declare the supported options.
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("iterations", po::value<int>(&test_iterations), "number of times, test is repeated")
        ("filename", po::value<std::string>(&filename)->multitoken(), "file from which column/haystack is imported")
        ("needle", po::value<std::string>(&needle)->multitoken(), "needle to search for")
        ("testcase", po::value<std::string>(&testname)->multitoken(), "testcase, can be substr or startswith")
        ("verbose", po::value<int>(&verbose)->implicit_value(1), "verbosity")
        ("exclude-gpu", po::value<int>(&use_gpu)->implicit_value(0), "exclude gpu tests")
        ("gpu-threads-per-block", po::value<int>(&threadnum), "Threads per Block on the GPU")
        ("measure-transfer", po::value<bool>(&measure_transfer)->implicit_value(true), "include the transfer time needed for the actual data in the measurements")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(ac, av, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
    }

    std::stringstream emptystream;
    std::ostream* otsts = &emptystream;
    if(verbose)
    	otsts = &std::cout;
    stringtest_interface* testcase;
    if(testname=="substr") {
        testcase = new substr_test(*otsts,measure_transfer);
    } else {
        testcase = new startswith_test(*otsts,measure_transfer);
    }

    std::vector<std::string> haystack;
    int minlen,maxlen,noChar;


    gpureset();
    read_haystack_varlen(filename.c_str(),haystack,minlen,maxlen,noChar);
    if(!haystack.size()) {
        exit(1);
    }

    std::cout<<"preparing data_structures: ";
    std::cout.flush();

    boost::timer::cpu_timer t_prepare;
    gpuStringArray gpu_haystack;
    gpu_haystack.init(haystack,noChar);

    dict_haystack cpu_d_haystack(haystack);
    std::cout<<t_prepare.format()<<std::endl;
    std::cout<<"haystack size: "<<haystack.size()<<", minimal string length: "<<minlen<<", maximum string length: "
            <<maxlen<<", unique:"<<cpu_d_haystack._dict.size/haystack.size()*100<<"%, size in MB: "<<(gpu_haystack.byte_size/1024/1024)<<std::endl;
    std::cout<<"number of iterations: "<<test_iterations<<", needle: "<<needle<<std::endl;

    *otsts<<"------------------------------"<<std::endl;

    omp_bug_workaround();

    std::vector<timestruct> times(20);
    for(int i = 0;i<test_iterations;++i) {
        uint testj=0;
        times[testj++].set_name("seq std")+=testcase->cpu_bench(haystack, needle.c_str());
        *otsts<<"------------------------------"<<std::endl;
        times[testj++].set_name("par std")+=testcase->cpu_bench_parallel(haystack, needle.c_str());
        *otsts<<"------------------------------"<<std::endl;
        testj++;

        times[testj++].set_name("seq custom")+=testcase->cpu_bench(gpu_haystack, needle.c_str());
        *otsts<<"------------------------------"<<std::endl;
        times[testj++].set_name("par custom")+=testcase->cpu_bench_parallel(gpu_haystack, needle.c_str());
        *otsts<<"------------------------------"<<std::endl;

        if(use_gpu) {
                times[testj++].set_name("gpu custom")+=testcase->gpu_bench(gpu_haystack, needle.c_str(),threadnum);
                *otsts<<"------------------------------"<<std::endl;
        } else  testj++;

        times[testj++].set_name("seq index")+=testcase->cpu_bench_with_index(gpu_haystack, needle.c_str());
        *otsts<<"------------------------------"<<std::endl;
        times[testj++].set_name("par index")+=testcase->cpu_bench_parallel_with_index(gpu_haystack, needle.c_str());
        *otsts<<"------------------------------"<<std::endl;

        if(use_gpu) {
                times[testj++].set_name("gpu index")+=testcase->gpu_bench_with_index(gpu_haystack, needle.c_str(),threadnum);
                *otsts<<"------------------------------"<<std::endl;
        } else  testj++;

        times[testj++].set_name("seq index/sse")+=testcase->cpu_bench_with_index_SSE(gpu_haystack, needle.c_str());
        *otsts<<"------------------------------"<<std::endl;
        times[testj++].set_name("par index/sse")+=testcase->cpu_bench_parallel_with_index_SSE(gpu_haystack, needle.c_str());
        *otsts<<"------------------------------"<<std::endl;
        testj++;

        times[testj++].set_name("seq dict")+=testcase->cpu_bench(cpu_d_haystack, needle.c_str());
        *otsts<<"------------------------------"<<std::endl;
        times[testj++].set_name("par dict")+=testcase->cpu_bench_parallel(cpu_d_haystack, needle.c_str());
        *otsts<<"------------------------------"<<std::endl;

        if(use_gpu) {
			times[testj++].set_name("gpu dict")+=testcase->gpu_bench(cpu_d_haystack, needle.c_str()/*,threadnum*/); //@todo threadnum
			*otsts<<"------------------------------"<<std::endl;
        } else  testj++;

        assert(testj<=times.size());
    }

    gpu_haystack.destroy();

#ifdef CSV_OUTPUT
    int i=1;
    for(std::vector<timestruct>::iterator it = times.begin();it<times.end();++it) {
        std::cout<<"{"<<it->test_name<<"} "<<*it/static_cast<float>(test_iterations)<<std::endl;
        if(!((i++)%3))
        	std::cout<<std::endl;
    }
#endif

    delete testcase;

    return 0;
}
