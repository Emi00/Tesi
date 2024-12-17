#include "utilities.h"


void generatePseudoSortedArray(double * v, int dim, int limit) {
    srand(time(NULL));rand();
    std::vector<int> toMove(dim,1);
    for(int i = 0 ; i < dim - limit; i++) {
        if(toMove[i]) {
            int idx = rand()%limit;
            std::swap(v[i],v[i+idx]);
            toMove[i] = 0;
            toMove[i+idx] = 0;
        }
    }
    for(int i = dim - 1; i > dim - limit ; i--) {
        std::swap(v[i],v[i-rand()%limit]);
    }
}


void debug(__m512i a) {
    int v[16];
    _mm512_storeu_epi32(v,a);
    std::cout<<"printv int: ";
    printv(v,16);
}

void debug64(__m512i a) {
    long v[8];
    _mm512_storeu_epi32(v,a);
    std::cout<<"printv int: ";
    printv(v,8);
}

void debug(__m512d a) {
    double v[8];
    _mm512_storeu_pd(v,a);
    std::cout<<"printv double: ";
    printv(v,8);
}

void debug(__mmask8 m) {
    for(int i = 0 ; i < 8 ; i++) {
        if((m>>i)&1) {
            std::cout<<"1 ";
        } else {
            std::cout<<"0 ";
        }
    }
    std::cout<<std::endl;
}

void printv(double * v, int n) {
    for(int i = 0 ; i< n ; i++) {
        std::cout<<v[i]<<" ";
    }
    std::cout<<std::endl;
}

void printv(int * v, int n) {
    for(int i = 0 ; i< n ; i++) {
        std::cout<<v[i]<<" ";
    }
    std::cout<<std::endl;
}


void printv(long * v, int n) {
    for(int i = 0 ; i< n ; i++) {
        std::cout<<v[i]<<" ";
    }
    std::cout<<std::endl;
}


bool isSortedAVX512_v1(double * v, int dim) {
    bool ans = 0;
    for(int i = 0 ; i < dim - 7 && !ans; i+=8) {
        __m512d arr1 = _mm512_loadu_pd(&v[i]);
        __m512d arr2 = _mm512_loadu_pd(&v[i + 1]);
        __mmask8 mask =_mm512_cmp_pd_mask(arr1,arr2,_CMP_GT_OS);
        ans |= (mask > 0);
    }
    for(int i = dim - (dim%8) + 1 ; i < dim && !ans; i++) {
        ans |= v[i] < v[i - 1];
    }
    return !ans;
}


void sort(double * v, int dim) {
    __m512d arr,min_vect;
    long idxs_arr[8] = {0,1,2,3,4,5,6,7};
    __m512i c,idxs_vect = _mm512_loadu_epi64(idxs_arr);
    __mmask8 mask;
    for(int i = 0 ; i < dim ; i++) {
        double minimum = v[i];
        double last = minimum;
        int idx = i;
        int j = i + 1;
        min_vect = _mm512_set1_pd(minimum);
        while(j < dim -8) {
            arr = _mm512_loadu_pd(&v[j]);
            mask = _mm512_cmplt_pd_mask(arr, min_vect);
            if(mask) {
                minimum = _mm512_reduce_min_pd(arr);
                min_vect = _mm512_set1_pd(minimum);
                mask = _mm512_cmpeq_pd_mask(arr,min_vect);
                c = _mm512_maskz_abs_epi64(mask,idxs_vect);
                idx = j + _mm512_reduce_max_epi64(c);
                min_vect = _mm512_set1_pd(minimum);
            }
            j+=8;
        }
        while(j < dim) {
            if(v[j] < v[idx]) {
                idx = j;
            }
            j++;
        }
        std::swap(v[i],v[idx]);
    }
}