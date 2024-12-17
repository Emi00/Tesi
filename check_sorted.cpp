#include "utils/timer.cpp"
#include <random>
#include <iostream>
#include <climits>
#include <immintrin.h>
#include <algorithm> 
#include <limits>

#pragma GCC target("avx512f,avx512dq,avx512cd,avx512bw,avx512vl,avx512vbmi,avx512ifma,avx512pf,avx512er,avx5124fmaps,avx5124vnniw,avx512bitalg,avx512vp2intersect")

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

bool isSorted_v1(double * v, int dim) {
    bool ans = 1;
    for(int i = 1 ; (i < dim) && ans ; i++) {
        ans &= (v[i-1] <= v[i]);
    }
    return ans;
}

bool isSortedAVX512_v1(double * v, int dim) {
    bool ans = 0;
    for(int i = 0 ; i < dim - 7 && !ans; i+=8) {
        __m512d arr1 = _mm512_loadu_pd(&v[i]);
        __m512d arr2 = _mm512_loadu_pd(&v[i + 1]);
        __mmask8 mask =_mm512_cmp_pd_mask(arr1,arr2,_CMP_GT_OS);
        //debug(arr1);
        //debug(arr2);
        //debug(mask);
        ans |= (mask > 0);
    }
    for(int i = dim - (dim%8) + 1 ; i < dim && !ans; i++) {
        ans |= v[i] < v[i - 1];
        //std::cout<<"cmp "<<(v[i] < v[i - 1])<<std::endl;
    }
    return !ans;
}



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


int main(int argn, char ** argv) {
    int alg = 0;
    if(argn >= 2) {
        alg = atoi(argv[1]);
    }
    int print = 0;
    if(argn >= 3) {
        print = atoi(argv[2]);
    }
    int order = 0;
    if(argn >= 4) {
        order = atoi(argv[3]);
    }
    int n = 10;
    if(order != 2 ) {
        if(argn >= 5) {
            n = atoi(argv[4]);
        }
    } else {
        n = argn - 4;
    }
    if(n <= 0) {
        return 0;
    }
    double v[n];
    if(order == 2) {
        for(int i = 0 ; i < n ; i++) {
            v[i] = atof(argv[i+4]);
        }
    } else {
        if(order == 1) {
            srand(time(NULL));rand();
            for(int i = 0 ; i < n ; i++) {
                v[i] = (double)rand();
            }
        } else {
            for(int i = 0 ; i < n ; i++) {
                v[i] = i;
            }
        }
    }
    if(order == 3) {
        generatePseudoSortedArray(v,n,10);
    } 
    if(print >= 2) {
        printv(v,n);
    }
    bool sorted = 0;
    switch(alg) {
        case 0:
            if(print) 
                std::cout<<"isSorted_v1"<<std::endl;
            {
                Timer t;
                sorted = isSorted_v1(v,n);
                t.stop();
            }
            break;
        case 1:
            if(print) 
                std::cout<<"isSortedAVX512_v1"<<std::endl;
            {
                Timer t;
                sorted = isSortedAVX512_v1(v,n);
                t.stop();
            }
            break;
        default:
            if(print) 
                std::cout<<"Algoritmo non trovato"<<std::endl;
    }
    if(print >= 2) {
        printv(v,n);
    }
    if(print) {
        bool expected = isSorted_v1(v,n);
        if(sorted == expected) {
            std::cout<<"worked"<<std::endl;
        } else {
            std::cout<<"did NOT work"<<std::endl;
        }
    }
}