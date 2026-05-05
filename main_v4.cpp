
#include <iostream>
#include <vector>
#include <chrono>
#include <immintrin.h>

// ! what if n % TILING_SIZE != 0 ????
const int TILING_SIZE = 128;

// matrix multiplication
void mulmat(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& c, int n) {
    for (int ii = 0; ii < n; ii += TILING_SIZE) {
        int i_limit = ii + TILING_SIZE;
    for (int kk = 0; kk < n; kk += TILING_SIZE) {
        int k_limit = kk + TILING_SIZE;
    for (int jj = 0; jj < n; jj += TILING_SIZE) {
        
        // assume index never over bound
        int j_limit = jj + TILING_SIZE;
        
        for (int i = ii; i < i_limit; i++) {
        for (int k = kk; k < k_limit; k++) {
            float r = a[i*n+k];
            __m512 v_a = _mm512_set1_ps(r);
            
            int j = jj;
            for (; j <= j_limit - 16; j += 16) { // vectorized
                // c[i*n+j] += r * b[k*n+j];
                __m512 v_c = _mm512_loadu_ps(&c[i*n+j]);
                __m512 v_b = _mm512_loadu_ps(&b[k*n+j]);
                
                // fma
                v_c = _mm512_fmadd_ps(v_a, v_b, v_c);
                // store v_res back to c[i*n+j]
                _mm512_storeu_ps(&c[i*n+j], v_c);
            } 
            // handle remaining elements
            for (; j < j_limit; j++) {
                c[i*n+j] += r * b[k*n+j];
            }
        } }
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
// 0: TILING_SIZE 32: 6s
// 1: TILING_SIZE 32: 6s
// 2: TILING_SIZE 32: 6s
// 3: TILING_SIZE 64: 7s
// 4: TILING_SIZE 64: 10s
// 5: TILING_SIZE 64: 6s
// 6: TILING_SIZE 128: 9s
// 7: TILING_SIZE 128: 8s
// 8: TILING_SIZE 128: 7s
