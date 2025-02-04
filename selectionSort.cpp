#include "utils/timer.cpp"
#include "utils/utilities.cpp"
#include <random>
#include <iostream>
#include <climits>
#include <immintrin.h>
#include <algorithm>
#include <limits>
using namespace std;
#pragma GCC target("avx512f,avx512dq,avx512cd,avx512bw,avx512vl,avx512vbmi,avx512ifma,avx512pf,avx512er,avx5124fmaps,avx5124vnniw,avx512bitalg,avx512vp2intersect")



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



void selectionSortAVX512_v1(double * v, int dim) {
    // declare the variable I'll be using
    __m512d vec;
    for(int i = 0 ; i < dim ; i++) {
        // set the variables I'll be using
        double minimum = v[i];
        double lastMin = minimum;
        int idx = i;
        int j = i + 1;
        // must not load longer than array dim, it MAY cause a SIGSEGV
        while(j <= dim - 8) {
            // load in the vec regiser the next 8 values
            vec = _mm512_loadu_pd(&v[j]);
            // set the minimum variable to the minimum between current min and next 8 values
            minimum = min(minimum,_mm512_reduce_min_pd(vec));
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

// almost fastest
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


// v5 + faster search for index of min
void selectionSortAVX512_v8(double * v, int dim) {
    __m512d vec,min_vect;
    __mmask8 mask;
    for(int i = 0 ; i < dim ; i++) {
        double minimum = v[i];
        double last = minimum;
        int idx = i;
        int j = i + 1;
        min_vect = _mm512_set1_pd(minimum);
        while(j < dim -8) {
            vec = _mm512_loadu_pd(&v[j]);
            mask = _mm512_cmplt_pd_mask(vec, min_vect);
            if(mask) {
                minimum = _mm512_reduce_min_pd(vec);
                min_vect = _mm512_set1_pd(minimum);
                mask = _mm512_cmpeq_pd_mask(vec,min_vect);
                idx = _tzcnt_u32(mask) + j;
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

// Chat-GPT o1
// A helper function to find the minimum value and its local index in a __m512d
// returning them as (minValue, localIndexIn0to7) via output parameters.
// localIndexIn0to7 = which lane among the 8 is the min.
static inline void horizontal_min8_with_index(__m512d v, double *outMinVal, int *outLaneIndex)
{
    // Store the 8 vector elements to a temp array
    double temp[8];
    _mm512_storeu_pd(temp, v);

    // Find the min and local index in scalar
    double minVal = temp[0];
    int minIdx = 0;
    for(int i = 1; i < 8; i++){
        if(temp[i] < minVal){
            minVal = temp[i];
            minIdx = i;
        }
    }
    *outMinVal = minVal;
    *outLaneIndex = minIdx;
}

// Chat-GPT o1
// Selection sort using AVX-512 to accelerate the "find minimum in arr[i+1..n-1]" step.
void selectionSortAVX512_ChatGPT_v2(double *arr, int n)
{
    for(int i = 0; i < n - 1; i++){
        double minVal = arr[i];
        int minIndex = i;

        // The inner loop: compare arr[j..j+7] in chunks with the current min
        __m512d vecMinVal = _mm512_set1_pd(minVal); // Broadcast current global min
        int jStart = i + 1;

        // Process in chunks of 8
        for(int j = jStart; j <= n - 8; j += 8){
            // Load next 8 elements
            __m512d data = _mm512_loadu_pd(&arr[j]);
            // Compare them to vecMinVal to find local minima
            // (branchless compare).
            __m512d maskMin = _mm512_min_pd(data, vecMinVal);

            // Now we want to see if maskMin is actually smaller than
            // the old vecMinVal in any lane. We'll do a horizontal reduction
            // to find the local min & lane index.
            double localMinVal;
            int localLaneIdx;
            horizontal_min8_with_index(maskMin, &localMinVal, &localLaneIdx);

            // If the local min is smaller than the global minVal, update
            // both minVal and minIndex
            if(localMinVal < minVal){
                minVal = localMinVal;
                minIndex = j + localLaneIdx;
                // Also refresh vecMinVal so future comparisons use the updated min
                vecMinVal = _mm512_set1_pd(minVal);
            }
        }

        // After the 8-wide loop, handle leftover elements if (n - jStart) not divisible by 8
        int leftoverStart = ((n - 1) / 8) * 8;
        if(leftoverStart < jStart) leftoverStart = jStart; // Just in case
        while(leftoverStart < n){
            if(arr[leftoverStart] < minVal){
                minVal = arr[leftoverStart];
                minIndex = leftoverStart;
            }
            leftoverStart++;
        }

        // Swap the found min element with arr[i], if needed
        if(minIndex != i){
            double tmp = arr[i];
            arr[i] = arr[minIndex];
            arr[minIndex] = tmp;
        }
    }
}


void selectionSortAVX512_DeepSeekr1_14b(double *arr, int n) {
    for (int i = 0; i < n; ++i) {
        // Find the minimum in the subarray from i to n-1
        int min_index = i;
        double min_val = arr[i];
        
        // Process the array in chunks of 8 elements
        for (int j = i + 8; j <= n; j += 8) {
            __m512d vec = _mm512_loadu_pd(arr + j - 8);
            
            // Compare each element in the vector to find the minimum
            int mask;
            __m512d miv_vec = _mm512_set1_pd(min_val);
            for (int k = 0; k < 8; ++k) {
               /* if (k == 0 || _mm512_cmp_pd_mask(
                    _mm512_shuffle_pd(vec, vec, (k << 4)), 
                    miv_vec, _CMP_LT_OS)) {
                    // Update the minimum value and index
                    //miv_vec = _mm512_shuffle_pd(vec, vec, (k << 4));
                    min_index = j - 8 + k;
                }*/
            }
        }
        
        // Swap the found minimum with the element at position i using vector operations
        if (min_index != i) {
            __m512d min_vec = _mm512_maskz_expandloadu_pd(0x01 << i,arr);
            __m512d current_vec = _mm512_loadu_pd(&arr[i]);
            
            // Replace the ith element with the minimum value
            _mm512_mask_compressstoreu_pd(arr, 0x01 << i, min_vec);
            
            // Shift all elements from min_index to i-1 one position to the right
            if (min_index > i) {
                _mm512_loadu_pd(&arr[i + 1]);
                //_mm512_storeu_pd(&arr[i],current_vec);
            }
        }
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
                selectionSort(v,n); // nella tesi
                t.stop();
            }
            break;
        case 1:
            if(print) 
                cout<<"selectionSortAVX512_v1"<<endl;
            {
                Timer t;
                selectionSortAVX512_v1(v,n); // nella tesi
                t.stop();
            }
            break;
        case 2:
            if(print) 
                cout<<"selectionSortAVX512_v2"<<endl;
            {
                Timer t;
                selectionSortAVX512_v2(v,n); // nella tesi
                t.stop();
            }
            break;
        case 3:
            if(print) 
                cout<<"selectionSortAVX512_copilot"<<endl;
            {
                Timer t;
                selectionSortAVX512_copilot(v,n); // nella tesi
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
        case 8:
            if(print) 
                cout<<"selectionSortAVX512_v8"<<endl;
            {
                Timer t;
                selectionSortAVX512_v8(v,n); // nella tesi
                t.stop();
            }
            break;
        case 9:
            if(print) 
                cout<<"selectionSortAVX512_ChatGPT_v2"<<endl;
            {
                Timer t;
                selectionSortAVX512_ChatGPT_v2(v,n); // nella tesi
                t.stop();
            }
            break;
        case 10:
            if(print) 
                cout<<"selectionSortAVX512_DeepSeekr1_14b"<<endl;
            {
                Timer t;
                selectionSortAVX512_DeepSeekr1_14b(v,n);
                t.stop();
            }
            break;
        case 11:
            if(print) 
                std::cout<<"std::sort"<<std::endl;
            {
                Timer t;
                sort(v,n);
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
            cout<<n<<" "<<n%8<<" sorted"<<endl;
        } else {
            cout<<n<<" "<<n%8<<" NOT sorted"<<endl;
        }
    }
}