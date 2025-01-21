#include <vector>
#include <iostream>

void add_vectors(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& result) {
    size_t n = a.size();
    
    #pragma omp simd
    for (size_t i = 0; i < n; ++i) {
        result[i] = a[i] + b[i];
    }
}

void selectionSort(double * v, int dim) {
    #pragma omp simd
    for(int i = 0 ; i < dim ; i++) {
        int idx = i;
        #pragma omp simd
        for(int j = i + 1 ; j < dim ; j++) {
            if(v[j] < v[idx]) {
                idx = j;
            }
        }
        std::swap(v[i],v[idx]);
    }
}

int main() {
    std::vector<float> a = {1.0, 2.0, 3.0, 4.0};
    std::vector<float> b = {5.0, 6.0, 7.0, 8.0};
    std::vector<float> result(4);

    add_vectors(a, b, result);

    for (float val : result) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}
