#include <iostream>
#include <vector>
#include <openblas/cblas.h> // openBLAS
#include <chrono>
#include <openblas/openblas_config.h>

int main() {
    openblas_set_num_threads(8);
    int num_threads = openblas_get_num_threads();
    std::cout << "Number of threads: " << num_threads << std::endl;

    std::srand(42);

    int n = 4096;
    std::vector<float> a(n * n, 1.0f);
    std::vector<float> b(n * n, 2.0f);
    std::vector<float> c(n * n, 0.0f);

    // fill a and b with random values
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            a[i*n+j] = std::rand() / (float)RAND_MAX;
            b[i*n+j] = std::rand() / (float)RAND_MAX;
        }
    }

    // C = alpha * A * B + beta * C
    float alpha = 1.0f;
    float beta = 0.0f;

    auto start = std::chrono::high_resolution_clock::now();

    cblas_sgemm(CblasRowMajor,  // layout：row major
                CblasNoTrans,   // A not transposed
                CblasNoTrans,   // B not transposed
                n, n, n,        // m, n, k size
                alpha,          // alpha 
                a.data(), n,    // A pointer and leading dimension
                b.data(), n,    // B pointer and leading dimension
                beta,           // beta 
                c.data(), n     // C pointer and leading dimension
    );

    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "Matrix Multiplication Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
    std::cout << "OpenBLAS Result Element: " << c[0] << std::endl;
    return 0;
}