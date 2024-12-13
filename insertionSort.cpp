#include "utils/timer.cpp"
#include <random>
#include <iostream>
#include <climits>

#pragma GCC target("avx512f,avx512dq,avx512cd,avx512bw,avx512vl,avx512vbmi,avx512ifma,avx512pf,avx512er,avx5124fmaps,avx5124vnniw,avx512bitalg,avx512vp2intersect")
#include <immintrin.h>
void insertionSort(int * v, int n) {
    for (int i = 1; i < n; ++i) {
        int key = v[i];
        int j = i - 1;
        
        while (j >= 0 && v[j] > key) {
            v[j + 1] = v[j];
            j = j - 1;
        }
        v[j + 1] = key;
    }
}


void printv(int * v, int n) {
    for(int i = 0 ; i< n ; i++) {
        std::cout<<v[i]<<" ";
    }
    std::cout<<std::endl;
}


int main(int argn, char ** argv) {
    int n = 10;
    if(argn >= 2) {
        n = atoi(argv[1]);
    }
    int m = 10;
    if(argn >= 3) {
        m = atoi(argv[2]);
    }
    int v[n];
    srand(time(NULL));rand();
    for(int i = 0 ; i < n ; ++i) {
        v[i] = rand()%m;
    }
    //printv(v,n);
    Timer t;
    insertionSort(v,n);
    t.stop();
    //printv(v,n);
}