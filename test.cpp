#include <vector>
#include <iostream>
#include <immintrin.h>


#pragma GCC target("avx512f")
void add_vectors(float * a, float * b, float * result, int n) {    
    #pragma omp simd avx512f
    #pragma vector always
    #pragma GCC ivdep
    for (int i = 0; i < n; ++i) {
        result[i] = a[i] + b[i];
    }
}


void vectorize_function(std::vector<float>& vec) {
    #pragma omp simd
    for (int i = 0; i < vec.size(); ++i) {
        vec[i] *= 2;
    }
}

//#pragma GCC target("avx512f")
void selectionSort(float * v, int dim) {
    //#pragma omp simd avx512f
    //#pragma vector always
    //#pragma GCC ivdep
    for(int i = 0 ; i < dim ; i++) {
        int idx = i;
        //#pragma omp simd avx512f
        //#pragma vector always
        //#pragma GCC ivdepsimd
        for(int j = i + 1 ; j < dim ; j++) {
            if(v[j] < v[idx]) {
                idx = j;
            }
        }
        std::swap(v[i],v[idx]);
    }
}

int main() {
    float a[16] = {1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0};
    float b[16] = {5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0};
    float result[16];

    add_vectors(a, b, result,16);

    selectionSort(result,16);

    //vectorize_function(result);

    for (int i = 0 ; i < 16 ; i++) {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
