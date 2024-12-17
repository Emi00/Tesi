#include <random>
#include <iostream>
#include <immintrin.h>

#define cicliRallenta 10000

void printv(double * v, int n);
void printv(int * v, int n);
void printv(long * v, int n);

void debug(__m512d a);
void debug(__m512i a);

bool isSortedAVX512_v1(double * v, int dim);

void generatePseudoSortedArray(double * v, int dim, int limit);