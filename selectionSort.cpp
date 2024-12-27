#include "utils/timer.cpp"
#include <random>
#include <iostream>
#include <climits>
#include <immintrin.h>
#include <algorithm>
#include <limits>
using namespace std;
#pragma GCC target("avx512f,avx512dq,avx512cd,avx512bw,avx512vl,avx512vbmi,avx512ifma,avx512pf,avx512er,avx5124fmaps,avx5124vnniw,avx512bitalg,avx512vp2intersect")

#define cicliRallenta 10000

void printv(double * v, int n);
void printv(int * v, int n);
void printv(long * v, int n);
void debug(__m512d a);
void debug(__m512i a);

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


// basic selectionSort, for comparasons
void selectionSort(double * v, int dim) {
    for(int i = 0 ; i < dim ; i++) {
        int idx = i;
        for(int j = i + 1 ; j < dim ; j++) {
            if(v[j] < v[idx]) {
                idx = j;
            }
        }
        swap(v[i],v[idx]);
    }
}


void debug(__m512i a) {
    int v[16];
    _mm512_storeu_epi32(v,a);
    cout<<"printv int: ";
    printv(v,16);
}

void debug64(__m512i a) {
    long v[8];
    _mm512_storeu_epi32(v,a);
    cout<<"printv int: ";
    printv(v,8);
}

void debug(__m512d a) {
    double v[8];
    _mm512_storeu_pd(v,a);
    cout<<"printv double: ";
    printv(v,8);
}

void debug(__mmask8 m) {
    for(int i = 0 ; i < 8 ; i++) {
        if((m>>i)&1) {
            cout<<"1 ";
        } else {
            cout<<"0 ";
        }
    }
    cout<<endl;
}



void selectionSortAVX512_v1(double * v, int dim) {
    // declare the variable I'll be using
    __m512d arr;
    for(int i = 0 ; i < dim ; i++) {
        // set the variables I'll be using
        double minimum = v[i];
        double lastMin = minimum;
        int idx = i;
        int j = i + 1;
        // must not load longer than array dim, it MAY cause a SIGSEGV
        while(j <= dim - 8) {
            // load in the arr regiser the next 8 values
            arr = _mm512_loadu_pd(&v[j]);
            // set the minimum variable to the minimum between current min and next 8 values
            minimum = min(minimum,_mm512_reduce_min_pd(arr));
            // if minimum is updated
            if(minimum != lastMin) {
                // set lastMin as current min
                lastMin = minimum;
                // find the index of the new found min
                for(int k = 0 ; k < 8 ; k++) {
                    if(v[j+k] == minimum) {
                        idx = j+k;
                        break;
                    }
                }
            }
            j += 8;
        }
        // use regular code to check the last values with index not multiple of 8
        while(j < dim) {
            if(v[j] < v[idx]) {
                idx = j;
            }
            j++;
        }
        // swap the values
        swap(v[i],v[idx]);
    }
}

void selectionSortAVX512_v2(double * v, int dim) {
    // declare the variables I'll be using
    __m512d arr,min_vect;
    // array of indexes, _mm512_maskz_mul_epi32 only considers alternated 32 bit of the register
    long idxs_arr[8] = {0,1,2,3,4,5,6,7};
    __m512i c,idxs_vect = _mm512_loadu_epi64(idxs_arr);
    __mmask8 mask;
    for(int i = 0 ; i < dim ; i++) {
        // set the variables I'll be using
        double minimum = v[i];
        double last = minimum;
        int idx = i;
        int j = i + 1;
        // must not load longer than array dim, it MAY cause a SIGSEGV
        while(j <= dim - 8) {
            //Timer t;
            //for(int ritardo = 0; ritardo < cicliRallenta ; ritardo++) {
            // load in the arr regiser the next 8 values
            arr = _mm512_loadu_pd(&v[j]);
            // set the minimum variable to the minimum between current min and next 8 values
            minimum = min(minimum,_mm512_reduce_min_pd(arr));
            //}
            //t.stop();
            // if minimum is updated
            if(minimum != last) { // there may be a better way to find the index of the current minimum, i'm trying to find it
                // set lastMin as current min
                //Timer t;
                //for(int ritardo = 0; ritardo < cicliRallenta ; ritardo++) {
                last = minimum;
                // set vect with minimum
                min_vect = _mm512_set1_pd(minimum);
                // get a mask with 1s only where the current minimum is
                mask = _mm512_cmpeq_pd_mask(arr,min_vect);
                // multiply the idxs_vect by 1 ONLY if the mask is set(e.g. if that index is of the current minimum)
                c = _mm512_maskz_abs_epi64(mask,idxs_vect);
                // update idx with max of the idx (each non zero value works)
                idx = j + _mm512_reduce_max_epi64(c);
                //}
                //t.stop();
            }
            j+=8;
        }
        // use regular code to check the last values with index not multiple of 8
        while(j < dim) {
            if(v[j] < v[idx]) {
                idx = j;
            }
            j++;
        }
        // swap the values
        swap(v[i],v[idx]);
    }
}



void selectionSortAVX512_copilot(double* arr, int n) {
    __m512d current_min_vals;
    __mmask8 mask;
    __m512d current_vals;
    for (int i = 0; i < n - 1; ++i) {
        int min_idx = i;
        double min_val = arr[i];

        for (int j = i + 1; j < n; j += 8) {
            //Timer t;
            //for(int ritardo = 0; ritardo < cicliRallenta ; ritardo++) {
            __m512d current_vals = _mm512_loadu_pd(&arr[j]);
            __m512d current_min_vals = _mm512_set1_pd(min_val);
            __mmask8 mask = _mm512_cmplt_pd_mask(current_vals, current_min_vals);
            //}
            //t.stop();
            if (mask) {
                //Timer t;
                //for(int ritardo = 0; ritardo < cicliRallenta ; ritardo++) {
                // Extract the new minimum value from current_vals
                for (int k = 0; k < 8; ++k) {
                    if (j + k < n && (mask & (1 << k))) {
                        if (arr[j + k] < min_val) {
                            min_val = arr[j + k];
                            min_idx = j + k;
                        }
                    }
                }
                //}
                //t.stop();
            }
        }

        // Swap the found minimum element with the first element
        if (min_idx != i) {
            std::swap(arr[i], arr[min_idx]);
        }
    }
}


void selectionSortAVX512_ChatGPT(double* arr, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        size_t min_index = i;
        double min_val = arr[i];

        // Find the minimum in the remaining array using AVX-512
        for (size_t j = i; j + 8 <= size; j += 8) {
            // Load 8 elements into an AVX-512 register
            __m512d reg = _mm512_loadu_pd(&arr[j]);

            // Compare and find the minimum value in the register
            __mmask8 cmp_mask = _mm512_cmp_pd_mask(reg, _mm512_set1_pd(min_val), _CMP_LT_OQ);

            // Update min_val and min_index directly using intrinsics
            if (cmp_mask) {
                // prende il primo, non funziona
                int first_true = _tzcnt_u32(cmp_mask); // Get the index of the first true comparison
                min_val = arr[j + first_true];
                min_index = j + first_true;
            }
        }

        // Check any remaining elements not fitting in an AVX-512 register
        for (size_t j = (size / 8) * 8; j < size; ++j) {
            if (arr[j] < min_val) {
                min_val = arr[j];
                min_index = j;
            }
        }

        // Swap the found minimum with the current element
        if (min_index != i) {
            std::swap(arr[i], arr[min_index]);
        }
    }
}

// best version
void selectionSortAVX512_v5(double * v, int dim) {
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
            }
            j+=8;
        }
        while(j < dim) {
            if(v[j] < v[idx]) {
                idx = j;
            }
            j++;
        }
        swap(v[i],v[idx]);
    }
}

// each iteration checks if it's already sorted; inefficient even with already substantially sorted
void selectionSortAVX512_v6(double * v, int dim) {
    __m512d arr,min_vect;
    long idxs_arr[8] = {0,1,2,3,4,5,6,7};
    __m512i c,idxs_vect = _mm512_loadu_epi64(idxs_arr);
    __mmask8 mask;
    for(int i = 0 ; i < dim ; i++) {
        if(isSortedAVX512_v1(&v[i],dim-i)) {
            break;
        }
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
        swap(v[i],v[idx]);
    }
}

// molto più lento di v6
void selectionSortAVX512_v7(double * v, int dim) {
    __m512d arr,min_vect,d;
    long idxs_arr[8] = {0,1,2,3,4,5,6,7};
    __m512i c,idxs_vect = _mm512_loadu_epi64(idxs_arr);
    __mmask8 mask;
    bool notSorted = 1;
    for(int i = 0 ; i < dim && notSorted; i++) {
        double minimum = v[i];
        double last = minimum;
        int idx = i;
        int j = i + 1;
        notSorted = 0;
        min_vect = _mm512_set1_pd(minimum);
        while(j < dim - 7) {
            arr = _mm512_loadu_pd(&v[j]);
            d = _mm512_loadu_pd(&v[j+1]);
            mask =_mm512_cmp_pd_mask(arr,d,_CMP_GT_OS);
            notSorted |= (mask > 0);
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
            notSorted |= v[j] < v[j-1];
            j++;
        }
        swap(v[i],v[idx]);
    }
}



void printv(double * v, int n) {
    for(int i = 0 ; i< n ; i++) {
        cout<<v[i]<<" ";
    }
    cout<<endl;
}

void printv(int * v, int n) {
    for(int i = 0 ; i< n ; i++) {
        cout<<v[i]<<" ";
    }
    cout<<endl;
}


void printv(long * v, int n) {
    for(int i = 0 ; i< n ; i++) {
        cout<<v[i]<<" ";
    }
    cout<<endl;
}


void generatePseudoSortedArray(double * v, int dim, int limit) {
    srand(time(NULL));rand();
    vector<int> toMove(dim,1);
    for(int i = 0 ; i < dim - limit; i++) {
        if(toMove[i]) {
            int idx = rand()%limit;
            swap(v[i],v[i+idx]);
            toMove[i] = 0;
            toMove[i+idx] = 0;
        }
    }
    for(int i = dim - 1; i > dim - limit ; i--) {
        swap(v[i],v[i-rand()%limit]);
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
    }
    v[n] = std::numeric_limits<double>::lowest();
    if(print >= 2) {
        printv(v,n);
    }
    switch(alg) {
        case 0:
            if(print) 
                cout<<"selectionSort"<<endl;
            {
                Timer t;
                selectionSort(v,n);
                t.stop();
            }
            break;
        case 1:
            if(print) 
                cout<<"selectionSortAVX512_v1"<<endl;
            {
                Timer t;
                selectionSortAVX512_v1(v,n);
                t.stop();
            }
            break;
        case 2:
            if(print) 
                cout<<"selectionSortAVX512_v2"<<endl;
            {
                Timer t;
                selectionSortAVX512_v2(v,n);
                t.stop();
            }
            break;
        case 3:
            if(print) 
                cout<<"selectionSortAVX512_copilot"<<endl;
            {
                Timer t;
                selectionSortAVX512_copilot(v,n);
                t.stop();
            }
            break;
        case 4:
            if(print) 
                cout<<"selectionSortAVX512_ChatGPT"<<endl;
            {
                Timer t;
                selectionSortAVX512_ChatGPT(v,n);
                t.stop();
            }
            break;
        case 5:
            if(print) 
                cout<<"selectionSortAVX512_v5"<<endl;
            {
                Timer t;
                selectionSortAVX512_v5(v,n);
                t.stop();
            }
            break;
        case 6:
            if(print) 
                cout<<"selectionSortAVX512_v6"<<endl;
            {
                Timer t;
                selectionSortAVX512_v6(v,n);
                t.stop();
            }
            break;
        case 7:
            if(print) 
                cout<<"selectionSortAVX512_v7"<<endl;
            {
                Timer t;
                selectionSortAVX512_v7(v,n);
                t.stop();
            }
            break;
        default:
            if(print) 
                cout<<"Algoritmo non trovato"<<endl;
    }
    if(print >= 2) {
        printv(v,n);
    }
    if(print) {
        selectionSort(tmp,n);
        bool ok = 1;
        for(int i = 0 ; i < n ; i++) {
            ok &= (v[i] == tmp[i]);
        }
        if(ok) {
            cout<<"sorted"<<endl;
        } else {
            cout<<"NOT sorted"<<endl;
        }
    }
}