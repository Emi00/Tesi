#include "utils/timer.cpp"
#include <random>
#include <iostream>
#include <climits>

#pragma GCC target("avx512f,avx512dq,avx512cd,avx512bw,avx512vl,avx512vbmi,avx512ifma,avx512pf,avx512er,avx5124fmaps,avx5124vnniw,avx512bitalg,avx512vp2intersect")
#include <immintrin.h>

#define cicliRallenta 10000

void bubbleSort(double * v, int n) {
    for(int i = n - 1 ; i >= 0 ; i--) {
        for(int j = 0 ; j < i ; j++) {
            if(v[j] > v[j+1]) {
                std::swap(v[j],v[j+1]);
            }
        }
    }
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


void debug(__m512i a) {
    int v[16];
    _mm512_store_epi32(v,a);
    std::cout<<"printv int: ";
    printv(v,16);
}

void debug64(__m512i a) {
    long v[8];
    _mm512_store_epi32(v,a);
    std::cout<<"printv int: ";
    printv(v,8);
}

void debug(__m512d a) {
    double v[8];
    _mm512_store_pd(v,a);
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

// per ogni passata sposto il valore massimo al massimo indice che ho visto
void bubbleSortAVX512_v1(double * v, int dim) {
    __m512d arr;
    for(int i = dim - 1 ; i >= 0 ; i--) {
        double maximum;
        int j = 0;
        while(j < i - 8) {
            arr = _mm512_loadu_pd(&v[j]);
            maximum = _mm512_reduce_max_pd(arr);
            if(maximum != v[j+7]) {
                //Timer t;
                //for(int ritardo = 0; ritardo < cicliRallenta ; ritardo++) {
                for(int k = 7 ; k >= 0 ; k--) {
                    if(v[j+k] == maximum) {
                        std::swap(v[j+7],v[j+k]);
                        break;
                    }
                }
                //}
                //t.stop();
            }
            j += 7;
        }
        while(j <= i - 1) {
            if(v[j] > v[j+1]) {
                std::swap(v[j],v[j+1]);
            }
            j++;
        }
    } 
    //t.stop();
}

// analogo a v1 ma con SIMD per non dover ciclare
void bubbleSortAVX512_v2(double * v, int dim) {
    __m512d arr,max_vect;
    long idxs_arr[16] = {0,1,2,3,4,5,6,7};
    __m512i c,idxs_vect = _mm512_load_epi64(idxs_arr);
    __mmask8 mask;
    for(int i = dim - 1 ; i >= 0 ; i--) {
        double maximum;
        int j = 0;
        while(j < i - 8) {
            arr = _mm512_loadu_pd(&v[j]);
            maximum = _mm512_reduce_max_pd(arr);
            if(maximum != v[j+7]) {
                //Timer t;
                //for(int ritardo = 0; ritardo < cicliRallenta ; ritardo++) {
                max_vect = _mm512_set1_pd(maximum);
                mask = _mm512_cmpeq_pd_mask(arr,max_vect);
                c = _mm512_maskz_abs_epi64(mask,idxs_vect);
                int idx = j + _mm512_reduce_max_epi64(c);
                std::swap(v[idx],v[j+7]);
                //}
                //t.stop();
            }
            j += 7;
        }
        while(j <= i - 1) {
            if(v[j] > v[j+1]) {
                std::swap(v[j],v[j+1]);
            }
            j++;
        }
    }
}


void bubbleSortAVX512_v3(double * v, int dim) {
    __m512d arr,max_vect;
    long idxs_arr[16] = {0,1,2,3,4,5,6,7};
    __m512i c,idxs_vect = _mm512_load_epi64(idxs_arr);
    __mmask8 mask;
    for(int i = dim - 1 ; i >= 0 ; i--) {
        double maximum;
        int j = 0;
        while(j < i - 8) {
            arr = _mm512_loadu_pd(&v[j]);
            maximum = _mm512_reduce_max_pd(arr);
            max_vect = _mm512_set1_pd(maximum);
            mask = _mm512_cmpeq_pd_mask(arr,max_vect);
            c = _mm512_maskz_abs_epi64(mask,idxs_vect);
            int idx = j + _mm512_reduce_max_epi64(c);
            std::swap(v[idx],v[j+7]);
            j += 7;
        }
        while(j <= i - 1) {
            if(v[j] > v[j+1]) {
                std::swap(v[j],v[j+1]);
            }
            j++;
        }
    }
}

int main(int argn, char ** argv) {
    int n = 8;
    // array dimension
    if(argn >= 2) {
        n = atoi(argv[1]);
    }
    // algorithm used
    int alg = 0;
    if(argn >= 3) {
        alg = atoi(argv[2]);
    }
    // do i have to print stuff?
    int print = 0;
    if(argn >= 4) {
        print = std::max(0,atoi(argv[3]));
    }
    int order = 0;
    if(argn >= 5) {
        order = atoi(argv[4]);
    }
    double v[n];
    double tmp[n];
    srand(time(NULL));rand();
    for(int i = 0 ; i < n ; ++i) {
        if(order == 0) {
            v[i] = (double)rand();
        } else if(order == 1){ // decreasing
            v[i] = n - i;
        } else { // increasing
            v[i] = i; 
        }
        tmp[i] = v[i];
    }
    if(print >= 2) {
        printv(v,n);
    }
    switch(alg) {
        case 0:
            if(print >= 1) 
                std::cout<<"bubbleSort"<<std::endl;
            {
                Timer t;
                bubbleSort(v,n);
                t.stop();
            }
            break;
        case 1:
            if(print >= 1) 
                std::cout<<"bubbleSortAVX512_v1"<<std::endl;
            {
                Timer t;
                bubbleSortAVX512_v1(v,n);
                t.stop();
            }
            break;
        case 2:
            if(print >= 1) 
                std::cout<<"bubbleSortAVX512_v2"<<std::endl;
            {
                Timer t;
                bubbleSortAVX512_v2(v,n);
                t.stop();
            }
            break;
        case 3:
            if(print >= 1) 
                std::cout<<"bubbleSortAVX512_v3"<<std::endl;
            {
                Timer t;
                bubbleSortAVX512_v3(v,n);
                t.stop();
            }
            break;
        default:
            if(print >= 1) 
                std::cout<<"Algoritmo non trovato"<<std::endl;
    }
    if(print >= 2) {
        printv(v,n);
    }
    if(print >= 1) {
        bubbleSort(tmp,n);
        bool ok = 1;
        for(int i = 0 ; i < n ; i++) {
            ok &= (v[i] == tmp[i]);
        }
        if(ok) {
            std::cout<<"sorted"<<std::endl;
        } else {
            std::cout<<"NOT sorted"<<std::endl;
        }
    }
}