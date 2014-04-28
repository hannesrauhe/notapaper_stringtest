#include "stringtest.hpp"
#include <algorithm>


std::ostream &operator<<(std::ostream &stream, const timestruct &t) {
    return stream << t.partime << " " << t.transfer << " " << t.seqtime << " " << t.overhead << " " << t.kernel;
}


bool gpuStringArray::strcmplt_cpu(const char* str1, const char* str2, uint l1, uint l2) {
    uint length=std::min(l1,l2);
    for(int i = 0; i<length; ++i) {
        if(str1[i]<str2[i]) {
            return true;
        } else if(str1[i]>str2[i]) {
            return false;
        }
        //continue on ==
    }
    return l1<l2;
}

bool gpuStringArray::strcmp2_cpu(const char* str1, const char* str2, uint length) {
//    uint length=std::min(l1,l2);
//    if(l1!=l2)
//        return false;
    for(int i = 0; i<length; ++i) {
        if(str1[i]!=str2[i]) {
            return false;
        }
    }
    return true;
}


int gpuStringArray::lower_bound ( int first, int last, const char* value, int length ) const {
    assert(first>=0);assert(first<=last);assert(last<=size);
    int it;
    int count, step;
    count = last-first;
    while (count>0)  {
        it = first; step=count/2; it+=step;
        if (strcmplt_cpu(&data[pos[it]],value,this->length[it],length)) {
            first=++it; count-=step+1;
        } else {
            count=step;
        }
    }
    return first;
}

int gpuStringArray::upper_bound ( int first, int last, const char* value, int length ) const {
    assert(first>=0);assert(first<=last);assert(last<=size);
    int it;
    int count, step;
    count = last-first;
    while (count>0)  {
        it = first; step=count/2; it+=step;
        if (!strcmplt_cpu(value,&data[pos[it]],length,this->length[it])) {
            first=++it; count-=step+1;
        } else {
            count=step;
        }
    }
    return first;
}

const char* gpuStringArray::operator [](int entryno) const {
    return &data[pos[entryno]];
}

dict_haystack::dict_haystack(const std::vector<std::string>& haystack) {
    std::vector<std::string> tdict(haystack);

    std::sort(tdict.begin(),tdict.end());
    std::vector<std::string>::iterator it = std::unique(tdict.begin(),tdict.end());

    tdict.resize(it - tdict.begin());

    _valueIDs.resize(haystack.size());
    #pragma omp parallel for
    for(int i = 0;i<haystack.size(); ++i ) {
        std::vector<std::string>::const_iterator hit = lower_bound(tdict.begin(),tdict.end(),haystack[i]);
        assert(hit!=tdict.end());
        assert(*hit==haystack[i]);
        _valueIDs[i]=hit-tdict.begin();
    }

    uint nochar = 0;
    for(std::vector<std::string>::const_iterator it = tdict.begin();it<tdict.end(); ++it ) {
        nochar+=it->length();
    }
    _dict.init(tdict,nochar);
}

dict_haystack::~dict_haystack() {
    _dict.destroy();
}


const char* dict_haystack::operator[](int pos) {
    assert(pos<_valueIDs.size());
    return _dict[_valueIDs[pos]];
}
