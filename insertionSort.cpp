#include "utils/timer.cpp"
#include <random>
#include <iostream>
#include <climits>
#include "utils/utilities.cpp"

#pragma GCC target("avx512f,avx512dq,avx512cd,avx512bw,avx512vl,avx512vbmi,avx512ifma,avx512pf,avx512er,avx5124fmaps,avx5124vnniw,avx512bitalg,avx512vp2intersect")
#include <immintrin.h>

int whileSISD = 0;
int whileSIMD = 0;
int ifSIMD = 0;

void insertionSort(double * v, int n) {
    for (int i = 1; i < n; ++i) {
        int key = v[i];
        int j = i - 1;
        while (j >= 0 && v[j] > key) {
            v[j + 1] = v[j];
            j--;
        }
        v[j + 1] = key;
    }
}



// NON FUNZIONA
void insertionSortAVX512_copilot(double* arr, int n) {
    for (int i = 1; i < n; ++i) {
        double key = arr[i];
        int j = i - 1;

        // Find the position where the key should be inserted
        while (j >= 0) {
            // Load 8 elements from the array
            __m512d vec = _mm512_loadu_pd(&arr[std::max(0, j - 7)]);

            // Compare the key with the elements
            __mmask8 mask = _mm512_cmplt_pd_mask(_mm512_set1_pd(key), vec);

            // Check if the key is less than any of the elements
            if (mask == 0) break;

            // Find the position to insert the key
            int pos = j;
            for (int k = 7; k >= 0; --k) {
                if (mask & (1 << k)) {
                    pos = j - (7 - k);
                    break;
                }
            }

            // Move the elements
            arr[j + 1] = arr[j];

            j = pos - 1;
        }
        arr[j + 1] = key;
    }
}

// idea: se il valore corrente è minore di 8 valori prima, carico 8 valori e li metto sfasati di 1 (posso usare la load e store unaligned)
void insertionSortAVX512_v1(double * v, int dim) {
    for (int i = 1; i < dim; ++i) {
        int j = i-1;
        double key = v[i];
        while(j >= 7 && v[j-7] > key) {
            whileSIMD++;
            __m512d vec = _mm512_loadu_pd(&v[j-7]);
            _mm512_storeu_pd(&v[j-6],vec);
            j -= 8;
        }
        while (j >= 0 && v[j] > key) {
            whileSISD++;
            v[j + 1] = v[j];
            j--;
        }
        v[j + 1] = key;
    }
}

// stesso concetto di v1 ma questa volta utilizzo funzioni SIMD anche se devo spostare meno di 8 posizioni
void insertionSortAVX512_v2(double * v, int dim) {
    __m512d current, vec, vecp1;
    __mmask8 mask;
    for (int i = 1; i < dim; ++i) {
        int j = i-1;
        double key = v[i];
        while(j >= 7 && v[j-7] > key) {
            whileSIMD++;
            vec = _mm512_loadu_pd(&v[j-7]);
            _mm512_storeu_pd(&v[j-6],vec);
            j -= 8;
        }
        if(j >= 7 && v[j-1] > key) {
            ifSIMD++;
            current = _mm512_set1_pd(key);
            vec = _mm512_loadu_pd(&v[j-7]);
            mask = _mm512_cmplt_pd_mask(current,vec);
            vecp1 = _mm512_loadu_pd(&v[j-6]);
            vec = _mm512_mask_blend_pd(mask,vecp1,vec);
            _mm512_storeu_pd(&v[j-6],vec);
            j-=8-_tzcnt_u32(mask);
        }
        while (j >= 0 && v[j] > key) {
            whileSISD++;
            v[j + 1] = v[j];
            j--;
        }
        v[j + 1] = key;
    }
}


// NON funziona, sistemabile rendendolo analogo a v1 ma con istruzioni in più come la set1, non ha senso
void insertionSortAVX512_ChatGPT(double* arr, size_t size) {
    for (size_t i = 1; i < size; i++) {
        double key = arr[i];  // Current element to insert
        size_t j = i;

        // Process in chunks of 8 using AVX-512
        while (j >= 8) {
            // Load 8 elements starting from position j-8
            __m512d vec = _mm512_loadu_pd(&arr[j - 8]);

            // Broadcast the key value across the entire AVX-512 register
            __m512d keyVec = _mm512_set1_pd(key);

            // Compare elements: mask out those greater than the key
            __mmask8 mask = _mm512_cmp_pd_mask(vec, keyVec, _CMP_GT_OQ);

            if (mask) {
                // Shift the 8 values one position to the right
                _mm512_storeu_pd(&arr[j - 7], vec);
                j -= 8;  // Move back by 8 positions
            } else {
                break;  // Stop shifting if no values are greater
            }
        }

        // Fallback to sequential shifting for remaining elements
        while (j > 0 && arr[j - 1] > key) {
            arr[j] = arr[j - 1];
            --j;
        }

        // Place the key in the correct position
        arr[j] = key;
    }
}

// come da aspettative è molto lento
void insertionSortAVX512_v3(double * v, int dim) {
    __m512d current, a,b;
    __mmask8 mask;
    for (int i = 1; i < dim; ++i) {
        int j = i-1;
        double key = v[i];
        while(j >= 7 && v[j-1] > key) {
            ifSIMD++;
            current = _mm512_set1_pd(key);
            a = _mm512_loadu_pd(&v[j-7]);
            mask = _mm512_cmplt_pd_mask(current,a);
            b = _mm512_loadu_pd(&v[j-6]);
            b = _mm512_mask_blend_pd(mask,b,a);
            _mm512_storeu_pd(&v[j-6],b);
            j-=8-_tzcnt_u32(mask);
        }
        while (j >= 0 && v[j] > key) {
            whileSISD++;
            v[j + 1] = v[j];
            j--;
        }
        v[j + 1] = key;
    }
}


int main(int argn, char ** argv) {
    int n = 8;
    if(argn >= 2) {
        n = atoi(argv[1]);
    }
    int alg = 0;
    if(argn >= 3) {
        alg = atoi(argv[2]);
    }
    int print = 0;
    if(argn >= 4) {
        print = atoi(argv[3]);
    }
    int order = 0;
    if(argn >= 5) {
        order = atoi(argv[4]);
    }
    double v[n+1];
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
    if(order == 3) {
        generatePseudoSortedArray(v,n,n/10);
        for(int i = 0 ; i < n ; i++) {
            tmp[i] = v[i];
        }
    }
    v[n] = std::numeric_limits<double>::lowest();
    if(print >= 2 && print != 3) {
        printv(v,n);
    }
    switch(alg) {
        case 0:
            if(print) 
                std::cout<<"insertionSort"<<std::endl;
            {
                Timer t;
                insertionSort(v,n); // nella tesi
                t.stop();
            }
            break;
        case 1:
            if(print) 
                std::cout<<"insertionSortAVX512_copilot"<<std::endl;
            {
                Timer t;
                insertionSortAVX512_copilot(v,n);
                t.stop();
            }
            break;
        case 2:
            if(print) 
                std::cout<<"insertionSortAVX512_v1"<<std::endl;
            {
                Timer t;
                insertionSortAVX512_v1(v,n); // nella tesi
                t.stop();
            }
            break;
        case 3:
            if(print) 
                std::cout<<"insertionSortAVX512_ChatGPT"<<std::endl;
            {
                Timer t;
                insertionSortAVX512_ChatGPT(v,n);
                t.stop();
            }
            break;
        case 4:
            if(print) 
                std::cout<<"insertionSortAVX512_v2"<<std::endl;
            {
                Timer t;
                insertionSortAVX512_v2(v,n); // nella tesi
                t.stop();
            }
            break;
        case 5:
            if(print) 
                std::cout<<"insertionSortAVX512_v3"<<std::endl;
            {
                Timer t;
                insertionSortAVX512_v3(v,n);
                t.stop();
            }
            break;
        default:
            if(print) 
                std::cout<<"Algoritmo non trovato"<<std::endl;
    }
    if(print >= 2 && print != 3) {
        printv(v,n);
    }
    if(print) {
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
        std::cout<<"whileSIMD: "<<whileSIMD<<std::endl;
        std::cout<<"ifSIMD: "<<ifSIMD<<std::endl;
        std::cout<<"whileSISD: "<<whileSISD<<std::endl;
    }
}