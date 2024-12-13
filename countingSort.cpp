#include "utils/timer.cpp"
#include <random>
#include <iostream>
#include <climits>


void printv(int * v, int n);

#pragma GCC target("avx512f,avx512dq,avx512cd,avx512bw,avx512vl,avx512vbmi,avx512ifma,avx512pf,avx512er,avx5124fmaps,avx5124vnniw,avx512bitalg,avx512vp2intersect")
#include <immintrin.h>
void countingSort(int * v, int n) {
    int a = v[0];
    int b = v[0];
    for(int i = 0 ; i < n ; i++) {
        a = std::max(a,v[i]);
        b = std::min(b,v[i]);
    }
    int counts[a-b+1];
    for(int i = 0 ; i < a-b+1 ; i++) {
        counts[i] = 0;
    }
    for(int i = 0 ; i < n ; i++) {
        counts[v[i]-b]++;
    }
    int idx = 0, i = 0;
    while(idx < a-b+1) {
        if(counts[idx] != 0) {
            v[i++] = idx+b;
            counts[idx]--;
        } else {
            idx++;
        }
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
    int m = 240000;
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
    countingSort(v,n);
    t.stop();
    //printv(v,n);
}