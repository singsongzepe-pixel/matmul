
#include <iostream>
#include <vector>
#include <chrono>

// basic matrix multiplication
void mulmat(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& c, int n) {
    for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
    for (int k = 0; k < n; k++) {
        c[i*n+j] += a[i*n+k] * b[k*n+j];
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

    return 0;
}

// test result:
// 0: 644 s
