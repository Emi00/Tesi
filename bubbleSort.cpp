#include "utils/timer.cpp"
#include <random>
#include <iostream>
#include <climits>
#include "utils/utilities.cpp"

//#pragma GCC target("avx512f,avx512dq,avx512cd,avx512bw,avx512vl,avx512vbmi,avx512ifma,avx512pf,avx512er,avx5124fmaps,avx5124vnniw,avx512bitalg,avx512vp2intersect")
#include <immintrin.h>

long long unsigned int cicli2 = 0;
long long unsigned int cicli1 = 0;
long long unsigned int cicli3 = 0;

void bubbleSort(double * v, int n) {
    for(int i = n - 1 ; i >= 0 ; i--) {
        for(int j = 0 ; j < i ; j++) {
            cicli1++;
            if(v[j] > v[j+1]) {
                cicli2++;
                std::swap(v[j],v[j+1]);
            }
        }
    }
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
            cicli1++;
            if(maximum != v[j+7]) {
                cicli2++;
                for(int k = 7 ; k >= 0 ; k--) {
                    if(v[j+k] == maximum) {
                        std::swap(v[j+7],v[j+k]);
                        break;
                    }
                }
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

// analogo a v1 ma con SIMD per non dover ciclare
void bubbleSortAVX512_v2(double * v, int dim) {
    __m512d arr,max_vect;
    long idxs_arr[16] = {0,1,2,3,4,5,6,7};
    __m512i c,idxs_vect = _mm512_loadu_epi64(idxs_arr);
    __mmask8 mask;
    for(int i = dim - 1 ; i >= 0 ; i--) {
        double maximum;
        int j = 0;
        while(j < i - 8) {
            arr = _mm512_loadu_pd(&v[j]);
            maximum = _mm512_reduce_max_pd(arr);
            cicli1++;
            if(maximum != v[j+7]) {
                cicli2++;
                max_vect = _mm512_set1_pd(maximum);
                mask = _mm512_cmpeq_pd_mask(arr,max_vect);
                c = _mm512_maskz_abs_epi64(mask,idxs_vect);
                int idx = j + _mm512_reduce_max_epi64(c);
                std::swap(v[idx],v[j+7]);
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

// senza if, sicuramente più lento ma il ciclo interno è branchless
void bubbleSortAVX512_v3(double * v, int dim) {
    __m512d arr,max_vect;
    long idxs_arr[16] = {0,1,2,3,4,5,6,7};
    __m512i c,idxs_vect = _mm512_loadu_epi64(idxs_arr);
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

// non ordina
void bubbleSortAVX512_copilot(double * v, int dim) {
    for(int i = 0 ; i < dim - 1; i++) {
        int swapped = 0;
        for(int j = 0 ; j < dim - 1 - i ; j += 8 ) {
            __m512d current = _mm512_loadu_pd(&v[j]);
            std::cout<<"curr"<<std::endl;
            debug(current);
            __m512d next = _mm512_loadu_pd(&v[j+1]);
            std::cout<<"next"<<std::endl;
            debug(next);
            __mmask8 mask = _mm512_cmp_pd_mask(current,next,_CMP_GT_OQ);
            std::cout<<"mask"<<std::endl;
            debug(mask);
            __m512d min = _mm512_mask_blend_pd(mask,current,next);
            std::cout<<"min"<<std::endl;
            debug(min);
            __m512d max = _mm512_mask_blend_pd(mask,next,current);
            std::cout<<"max"<<std::endl;
            debug(max);
            _mm512_storeu_pd(&v[j],min);
            _mm512_storeu_pd(&v[j+1],max);
            printv(&v[j],9);
        }
    }
}

// non ordina
void bubbleSortAVX512_ChatGPT(double* array, std::size_t dim) {
    if (!array || dim < 2) return; // Controllo dei limiti

    const std::size_t simd_width = 8; // AVX512 lavora su 8 double contemporaneamente

    // Iterazioni esterne del Bubble Sort
    for (std::size_t i = 0; i < dim - 1; ++i) {
        bool swapped = false;

        // Elaborazione in blocchi SIMD
        for (std::size_t j = 0; j + simd_width <= dim - i - 1; j += simd_width) {
            // Carica 8 double consecutivi
            __m512d current = _mm512_loadu_pd(&array[j]);
            __m512d next = _mm512_loadu_pd(&array[j + 1]);

            // Confronta gli elementi e scambia se necessario
            __mmask8 mask = _mm512_cmp_pd_mask(current, next, _CMP_GT_OQ);
            if (mask) {
                swapped = true;
                // Scambia gli elementi
                __m512d swapped_current = _mm512_mask_blend_pd(mask, current, next);
                __m512d swapped_next = _mm512_mask_blend_pd(mask, next, current);

                // Memorizza i risultati
                _mm512_storeu_pd(&array[j], swapped_current);
                _mm512_storeu_pd(&array[j + 1], swapped_next);
            }
        }

        // Gestione della coda non allineata ai blocchi SIMD
        for (std::size_t j = dim - i - 1 - (dim - i - 1) % simd_width; j < dim - i - 1; ++j) {
            if (array[j] > array[j + 1]) {
                swapped = true;
                std::swap(array[j], array[j + 1]);
            }
        }

        // Se nessuno scambio è stato effettuato, l'array è ordinato
        if (!swapped) break;
    }
}

void bubbleSortAVX512_v4(double * v, int dim) {
    __m512d arr, curr;
    __mmask8 mask;
    for(int i = dim - 1 ; i >= 0 ; i--) {
        int j = 0;
        int maximum = v[j];
        curr = _mm512_set1_pd(v[j]);
        while(j < i - 8) {
            cicli1++;
            arr = _mm512_loadu_pd(&v[j]);
            mask = _mm512_cmplt_pd_mask(curr,arr);
            int idx = j;
            if(mask) {
                cicli2++;
                for(int k = idx ; k < j+8 ; k++) {
                    if(v[k] > maximum) {
                        idx = k;
                        maximum = v[k];
                    }
                }
            }
            j+=7;
            std::swap(v[j],v[idx]);
        }
        while(j <= i - 1) {
            if(v[j] > v[j+1]) {
                std::swap(v[j],v[j+1]);
            }
            j++;
        }
    }
}

void bubbleSortAVX512_v5(double * v, int dim) {
    __m512d arr, curr;
    __mmask8 mask;
    for(int i = dim - 1 ; i >= 0 ; i--) {
        int j = 0;
        double maximum = v[j];
        while(j < i - 8) {
            cicli1++;
            arr = _mm512_loadu_pd(&v[j]);
            curr = _mm512_set1_pd(v[j+7]);
            mask = _mm512_cmplt_pd_mask(curr,arr);
            if(mask) {
                maximum = _mm512_reduce_max_pd(arr);
                cicli2 += (v[j]==maximum);
                curr = _mm512_set1_pd(maximum);
                mask = _mm512_cmpeq_pd_mask(arr,curr);
                int idx = _tzcnt_u32(mask) + j;
                std::swap(v[j+7],v[idx]);
            }
            j+=7;
        }
        while(j <= i - 1) {
            if(v[j] > v[j+1]) {
                std::swap(v[j],v[j+1]);
            }
            j++;
        }
    }
}

// best version
void bubbleSortAVX512_v6(double * v, int dim) {
    __m512d arr, curr;
    __mmask8 mask;
    for(int i = dim - 1 ; i >= 0 ; i--) {
        int j = 1;
        curr =  _mm512_set1_pd(v[0]);
        while(j < i - 8) {
            arr = _mm512_loadu_pd(&v[j]);
            mask = _mm512_cmplt_pd_mask(curr,arr);
            if(!mask) {
                cicli2++;
                std::swap(v[j+7],v[j-1]);
            } else {
                cicli1++;
                curr = _mm512_set1_pd(v[j+7]);
                mask = _mm512_cmplt_pd_mask(curr,arr);
                if(mask) {
                    cicli3++;
                    double maximum = _mm512_reduce_max_pd(arr);
                    curr = _mm512_set1_pd(maximum);
                    mask = _mm512_cmpeq_pd_mask(arr,curr);
                    int idx = _tzcnt_u32(mask) + j;
                    std::swap(v[j+7],v[idx]);
                }
            }
            j+=8;
        }
        while(j <= i ) {
            if(v[j-1] > v[j]) {
                std::swap(v[j-1],v[j]);
            }
            j++;
        }
    }
}

void bubbleSortAVX512_v7(double * v, int dim) {
    __m512d arr, curr;
    __mmask8 mask;
    for(int i = dim - 1 ; i >= 0 ; i--) {
        int j = 0;
        while(j < i - 8) {
            arr = _mm512_loadu_pd(&v[j]);
            curr = _mm512_set1_pd(v[j+7]);
            mask = _mm512_cmplt_pd_mask(curr,arr);
            cicli1++;
            if(mask) {
                cicli2++;
                double maximum = _mm512_reduce_max_pd(arr);
                curr = _mm512_set1_pd(maximum);
                mask = _mm512_cmpeq_pd_mask(arr,curr);
                int idx = _tzcnt_u32(mask) + j;
                std::swap(v[j+7],v[idx]);
            }
            j+=7;
        }
        while(j < i ) {
            if(v[j] > v[j+1]) {
                std::swap(v[j+1],v[j]);
            }
            j++;
        }
    }
}


void bubbleSortAVX512_v8(double * v, int dim) {
    __m512d arr, curr;
    __mmask8 mask;
    for(int i = dim - 1 ; i >= 0 ; i--) {
        int j = 1;
        curr =  _mm512_set1_pd(v[0]);
        while(j < i - 8) {
            arr = _mm512_loadu_pd(&v[j]);
            mask = _mm512_cmplt_pd_mask(curr,arr);
            cicli1++;
            if(!mask) {
                cicli2++;
                std::swap(v[j+7],v[j-1]);
            } else {
                cicli3++;
                curr = _mm512_set1_pd(v[j+7]);
                mask = _mm512_cmplt_pd_mask(curr,arr);
                double maximum = _mm512_reduce_max_pd(arr);
                curr = _mm512_set1_pd(maximum);
                mask = _mm512_cmpeq_pd_mask(arr,curr);
                int idx = _tzcnt_u32(mask) + j;
                std::swap(v[j+7],v[idx]);
            }
            j+=8;
        }
        while(j <= i ) {
            if(v[j-1] > v[j]) {
                std::swap(v[j-1],v[j]);
            }
            j++;
        }
    }
}

void bubbleSortAVX512_v9(double * v, int dim) {
    __m512d arr, curr;
    __mmask8 mask;
    for(int i = dim - 1 ; i >= 0 ; i--) {
        int j = 1;
        curr =  _mm512_set1_pd(v[0]);
        while(j < i - 8) {
            arr = _mm512_loadu_pd(&v[j]);
            mask = _mm512_cmplt_pd_mask(curr,arr);
            cicli1++;
            if(!mask) {
                cicli2++;
                std::swap(v[j+7],v[j-1]);
            } else {
                curr = _mm512_set1_pd(v[j+7]);
                mask = _mm512_cmplt_pd_mask(curr,arr);
                if(mask) {
                    for(int k = 0 ; k < 7 ; k++) {
                        curr = _mm512_set1_pd(v[j+k]);
                        mask = _mm512_cmplt_pd_mask(curr,arr);
                        cicli3++;
                        if(!mask) {
                            std::swap(v[j+7],v[j+k]);
                            break;
                        }
                    }
                }
            }
            j+=8;
        }
        while(j <= i ) {
            if(v[j-1] > v[j]) {
                std::swap(v[j-1],v[j]);
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
        } else if(order == 2){ // increasing
            v[i] = i;
        } else {
            v[i] = (double)(rand()%n);
        }
        tmp[i] = v[i];
    }
    if(order == 3) {
        generatePseudoSortedArray(v,n,n/10);
        for(int i = 0 ; i < n ; i++) {
            tmp[i] = v[i];
        }
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
                bubbleSort(v,n); // nella tesi
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
        case 4:
            if(print >= 1) 
                std::cout<<"bubbleSortAVX512_copilot"<<std::endl;
            {
                Timer t;
                bubbleSortAVX512_copilot(v,n);
                t.stop();
            }
            break;
        case 5:
            if(print >= 1) 
                std::cout<<"bubbleSortAVX512_ChatGPT"<<std::endl;
            {
                Timer t;
                bubbleSortAVX512_ChatGPT(v,n);
                t.stop();
            }
            break;
        case 6:
            if(print >= 1) 
                std::cout<<"bubbleSortAVX512_v4"<<std::endl;
            {
                Timer t;
                bubbleSortAVX512_v4(v,n);
                t.stop();
            }
            break;
        case 7:
            if(print >= 1) 
                std::cout<<"bubbleSortAVX512_v5"<<std::endl;
            {
                Timer t;
                bubbleSortAVX512_v5(v,n); // nella tesi
                t.stop();
            }
            break;
        case 8:
            if(print >= 1) 
                std::cout<<"bubbleSortAVX512_v6"<<std::endl;
            {
                Timer t;
                bubbleSortAVX512_v6(v,n); // nella tesi
                t.stop();
            }
            break;
        case 9:
            if(print >= 1) 
                std::cout<<"bubbleSortAVX512_v7"<<std::endl;
            {
                Timer t;
                bubbleSortAVX512_v7(v,n);
                t.stop();
            }
            break;
        case 10:
            if(print >= 1) 
                std::cout<<"bubbleSortAVX512_v8"<<std::endl;
            {
                Timer t;
                bubbleSortAVX512_v8(v,n);
                t.stop();
            }
            break;
        case 11:
            if(print >= 1) 
                std::cout<<"bubbleSortAVX512_v9"<<std::endl;
            {
                Timer t;
                bubbleSortAVX512_v9(v,n);
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
        sort(tmp,n);
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
    if(print >= 3) {
        std::cout<<"cicli1: "<<cicli1<<" cicli2: "<<cicli2<<" cicli3: "<<cicli3<<std::endl;
    }
}