
#include <iostream>
#include <vector>
#include <chrono>
#include <immintrin.h>


template <typename T, std::size_t Alignment>
struct CacheAlignedAllocator {
    using value_type = T;
    CacheAlignedAllocator() noexcept = default;

    template <typename U>
    CacheAlignedAllocator(const CacheAlignedAllocator<U, Alignment>&) noexcept {}
    
    template <typename U> struct rebind { using other = CacheAlignedAllocator<U, Alignment>; };

    T* allocate(std::size_t n) {
        if (n == 0) return nullptr;
        void* ptr = nullptr;
#if defined(_WIN32) || defined(__MINGW32__)
        ptr = _aligned_malloc(n * sizeof(T), Alignment);
#else 
        if (posix_memalign(&ptr, Alignment, n * sizeof(T)) != 0) ptr = nullptr;
#endif
        if (!ptr) throw std::bad_alloc();
        return static_cast<T*>(ptr);
    }

    void deallocate(T* ptr, std::size_t n) {
#if defined(_WIN32) || defined(__MINGW32__)
        _aligned_free(ptr);
#else 
        free(ptr);
#endif  
    }   
    
    bool operator==(const CacheAlignedAllocator&) const { return true; }
    bool operator!=(const CacheAlignedAllocator&) const { return false; }
};

template <typename T>
using CacheAlignedVector = std::vector<T, CacheAlignedAllocator<T, 64>>;

// ! what if n % TILING_SIZE != 0 ????
const int TILING_SIZE = 64;

/* this is why we need stride_offset
# 这里有一个非常容易出现的问题，就是
# 现代的cache大部分都是组相联设计的
# 完美的对齐反而容易被映射到同一组中
# 我们对传入的内存对齐的矩阵进行一些偏移
# 可以避免这种情况发生，效率几乎提升一倍
# 来到1121ms左右

# there is also a interesting thing,
# if you change stride_offset a little bit,
# like change it to 15,
# then the efficiency will decrease to about 2723 ms
# because the the 512-bit data for avx2 instruction is not aligned to 64 bytes
# they are not in the same cache line
*/
const int stride_offset = 16;

// matrix multiplication
void mulmat(const CacheAlignedVector<float>& a, const CacheAlignedVector<float>& b, CacheAlignedVector<float>& c, int n) {
    int stride = n + stride_offset;

    #pragma omp parallel for collapse(2) schedule(dynamic, 64) // flatten ii, jj into a bigger task pool
    for (int ii = 0; ii < n; ii += TILING_SIZE) {
    for (int jj = 0; jj < n; jj += TILING_SIZE) {
        int i_limit = ii + TILING_SIZE;
        int j_limit = jj + TILING_SIZE;

        // (ii, jj) is unique, and their c block won't intersect
        for (int kk = 0; kk < n; kk += TILING_SIZE) {
            int k_limit = kk + TILING_SIZE;
            
            for (int i = ii; i < i_limit; i++) {
            for (int k = kk; k < k_limit; k++) {
                float r = a[i*stride+k];
                __m512 v_a = _mm512_set1_ps(r);
                
                int j = jj;
                for (; j <= j_limit - 16; j += 16) { // vectorized
                    // c[i*n+j] += r * b[k*n+j];
                    __m512 v_c = _mm512_load_ps(&c[i*stride+j]);
                    __m512 v_b = _mm512_load_ps(&b[k*stride+j]);
                    
                    // fma
                    v_c = _mm512_fmadd_ps(v_a, v_b, v_c);
                    // store v_res back to c[i*n+j]
                    _mm512_store_ps(&c[i*stride+j], v_c);
                } 
                // handle remaining elements
                for (; j < j_limit; j++) {
                    c[i*stride+j] += r * b[k*stride+j];
                }
            } } 
        } 
    } }
}

int main() 
{
    std::srand(42);

    // a is 4096x4096 matrix
    int n = 4096;
    int stride = n + stride_offset;
    CacheAlignedVector<float> a(n * stride);
    CacheAlignedVector<float> b(n * stride);
    CacheAlignedVector<float> c(n * stride);
    // random value should fall in [0, 1.0f)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            a[i*stride+j] = std::rand() / (float)RAND_MAX;
            b[i*stride+j] = std::rand() / (float)RAND_MAX;
        }
    }
    
    // measure time
    auto start = std::chrono::high_resolution_clock::now();

    mulmat(a, b, c, n);
    
    auto end = std::chrono::high_resolution_clock::now();
    // time in seconds
    std::cout << "Matrix Multiplication Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
    
    // check the result
    std::cout << "Check the Result at (0, 0):" << c[0] << std::endl;

    return 0;
}

// test result:
// 0: TILING_SIZE 64: 1112 ms
// 1: TILING_SIZE 64: 1178 ms
// 2: TILING_SIZE 64: 1321 ms
// 3: TILING_SIZE 64: 1579 ms
// 4: TILING_SIZE 64: 1100 ms


