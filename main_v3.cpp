
#include <iostream>
#include <vector>
#include <chrono>
#include <immintrin.h>

// ! what if n % TILING_SIZE != 0 ????
const int TILING_SIZE = 64;

// matrix multiplication
void mulmat(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& c, int n) {
    for (int ii = 0; ii < n; ii += TILING_SIZE) {
    for (int kk = 0; kk < n; kk += TILING_SIZE) {
    for (int jj = 0; jj < n; jj += TILING_SIZE) {
        for (int i = ii; i < ii + TILING_SIZE; i++) {
        for (int k = kk; k < kk + TILING_SIZE; k++) {
            float r = a[i * n + k];
        for (int j = jj; j < jj + TILING_SIZE; j++) {
            c[i * n + j] += r * b[k * n + j];
        } } }
    } } }
}

int main() 
{
    std::srand(42);

    // a is 4096x4096 matrix
    int n = 4096;
    std::vector<float> a(n * n);
    std::vector<float> b(n * n);
    std::vector<float> c(n * n);
    // random value should fall in [0, 1.0f)
    for (int i = 0; i < n * n; i++) {
        a[i] = std::rand() / (float)RAND_MAX;
        b[i] = std::rand() / (float)RAND_MAX;
    }
    
    // measure time
    auto start = std::chrono::high_resolution_clock::now();

    mulmat(a, b, c, n);
    
    auto end = std::chrono::high_resolution_clock::now();
    // time in seconds
    std::cout << "Matrix Multiplication Time: " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << " s" << std::endl;
    
    // randomly check the result
    std::cout << "Randomly Check the Result:" << c[std::rand() % (n*n)] << std::endl;

    return 0;
}

// test result:
// 0: Tiling_SIZE 32: 22 s
// 1: Tiling_SIZE 32: 17 s
// 2: Tiling_SIZE 64: 16 s
// 3: Tiling_SIZE 64: 16 s
// 4: Tiling_SIZE 128: 17 s
// 5: Tiling_SIZE 128: 14 s
// 6: Tiling_SIZE 16: 32 s
// 7: Tiling_SIZE 256: 17 s
