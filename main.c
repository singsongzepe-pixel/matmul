
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// a, b, c are all n x n matrix
void matmul(float *restrict a, float *restrict b, float *restrict c, int n) {
    for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
    for (int k = 0; k < n; k++) {
        c[i*n+j] += a[i*n+k] * b[k*n+j];
    } } }
}

int main()
{
    // a is 4096x4096 matrix
    int n = 4096;
    float *a = (float*)malloc(n * n * sizeof(float));
    float *b = (float*)malloc(n * n * sizeof(float));

    float *c = (float*)malloc(n * n * sizeof(float));

    // fill a and b with random values
    // random value should fall in [0, 1.0f)
    for (int i = 0; i < n * n; i++) {
        a[i] = rand() / (float)RAND_MAX;
        b[i] = rand() / (float)RAND_MAX;
    }

    // measure time
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    matmul(a, b, c, n);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_spent = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("Result: %f, Time: %f s\n", c[rand() % (n*n)], time_spent);

    // free memory
    free(a);
    free(b);
    free(c);

    return 0;
}

// test result:
// 0: 773.053014 s
// 1: 648.161988 s