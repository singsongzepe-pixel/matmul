

#include <iostream>
#include <vector>
#include <chrono>
#include <immintrin.h>
#include <algorithm>
#include <omp.h>

// #define DBUG
#define COMPARE

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
        if (!ptr) std::exit(-5);
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

template <typename T>
CacheAlignedVector<T> from(const std::vector<T>& orig) {
    return CacheAlignedVector<T>(orig.begin(), orig.end());
}

// some magic number for Tiger Lake
// const int Mc = 1024;
const int Kc = 256;
const int Nc = 4096;
const int mr = 16;
const int nr = 16;

__attribute__((always_inline))
inline void packMatrixB(
    const float *__restrict__ b,
    float *__restrict__ b_panel,
    int bi, int bj, int n
) {
    float *dest_ptr = b_panel;

    for (int j = 0; j < Nc; j += nr) {
        for (int k = 0; k < Kc; k++) {
            const float *src_row_ptr = b + (bi + k)*n + (bj + j);

            std::copy_n(src_row_ptr, nr, dest_ptr);
            dest_ptr += nr;
        }
    }
}

__attribute__((always_inline))
inline void packMatrixA(
    const float *__restrict__ a,
    float *__restrict__ a_panel,
    int ai, int aj, int Mc, int n
) {
    float *dest_ptr = a_panel;

    for (int i = 0; i < Mc; i += mr) {  
        for (int k = 0; k < Kc; k++) {
            const float *src_row_ptr = a + (ai + i)*n + (aj + k);
            
            for (int r = 0; r < mr; r++) {
                *dest_ptr++ = src_row_ptr[r*n];
            }
        }
    }
}

// void micro_kernel_16x16(
//     const float* a_panel, // 16 x K
//     const float* b_panel, // K x 16
//     float* c,             // 16x16
//     int ldc,              // n (num_column) for fetching the corresponding c address on the other row
//     int K                 // Kc
// ) {
//     __m512 c_reg[16];
//     for (int i = 0; i < 16; ++i) {
//         // c is not 64 aligned
//         c_reg[i] = _mm512_load_ps(c + i * ldc);
//     }

//     for (int k = 0; k < K; ++k) {
//         // a_panel and b_panel is 64 aligned
//         __m512 b_row = _mm512_loadu_ps(b_panel + k * 16);

//         const float* a_col_ptr = a_panel + k * 16;

//         // unrolling
//         for (int i = 0; i < 16; ++i) {
//             __m512 a_val = _mm512_set1_ps(a_col_ptr[i]);
            
//             // fma
//             c_reg[i] = _mm512_fmadd_ps(a_val, b_row, c_reg[i]);
//         }
//     }

//     for (int i = 0; i < 16; ++i) {
//         _mm512_storeu_ps(c + i * ldc, c_reg[i]);
//     }
// }

extern "C" void micro_kernel_16x16(
    const float *a_panel,
    const float *b_panel,
    float *c,
    int ldc,
    int K
);


#ifdef DBUG
__attribute__((always_inline))
inline void packMatrixB_2(
    const std::vector<float>& b,
    CacheAlignedVector<float>& b_panel,
    int bi, int bj, int n 
) {
    for (int i = 0; i < Kc; i++) {      
        for (int j = 0; j < Nc; j++) {   
            if ((bi + i) < n && (bj + j) < n) {
                b_panel[i * Nc + j] = b[(bi + i) * n + (bj + j)];
            } else {
                b_panel[i * Nc + j] = 0.0f; 
            }
        }
    }
}
#endif

// matrix multiplication
void mulmat(
    const std::vector<float>& a,
    const std::vector<float>& b,
    std::vector<float>& c,
    int n
) { 
    int max_threads = omp_get_max_threads();
    int target_tasks_per_thread = 4;
    int total_target_tasks = max_threads * target_tasks_per_thread;

    int Mc = (n / total_target_tasks) & ~(mr - 1);
    if (Mc < 64) Mc = 64;
    if (Mc > 1024) Mc = 1024;

    // std::cout << "Mc: " << Mc << std::endl;

    CacheAlignedVector<float> b_panel(Kc * Nc, 0.0f);
    std::vector<CacheAlignedVector<float>> a_panels(max_threads, CacheAlignedVector<float>(Mc * Kc, 0.0f));

    // for LLC, make B into panels of size Kc x Nc (4MB)
    for (int jc = 0; jc < n; jc += Nc) {
        for (int pc = 0; pc < n; pc += Kc) {

            // put a panel of b into b_panel and also in LLC
            packMatrixB(b.data(), b_panel.data(), pc, jc, n);
            
            // for L2, make A into panels of size Mc x Kc (1MB)
            #pragma omp parallel for schedule(dynamic)
            for (int ic = 0; ic < n; ic += Mc) {
                int tid = omp_get_thread_num();
                
                int current_Mc = std::min(Mc, n - ic);
                packMatrixA(a.data(), a_panels[tid].data(), ic, pc, current_Mc, n);

                for (int jr = 0; jr < Nc; jr += nr) {
                    for (int ir = 0; ir < current_Mc; ir += mr) {
                        const float* current_a_panel = a_panels[tid].data() + ir * Kc;
                        const float* current_b_panel = b_panel.data() + jr * Kc;
                        
                        float* c_ptr = c.data() + (ic + ir) * n + (jc + jr);

                        micro_kernel_16x16(current_a_panel, current_b_panel, c_ptr, n, Kc);
                    }
                }
            }
        }
    }
}

#ifdef COMPARE
// matrix multiplication
const int TILING_SIZE = 64;

// matrix multiplication
void mulmat_v6(
    const CacheAlignedVector<float>& a, 
    const CacheAlignedVector<float>& b, 
    CacheAlignedVector<float>& c, 
    int n
) {
    #pragma omp parallel for collapse(2) // flatten ii, jj into a bigger task pool
    for (int ii = 0; ii < n; ii += TILING_SIZE) {
    for (int jj = 0; jj < n; jj += TILING_SIZE) {
        int i_limit = ii + TILING_SIZE;
        int j_limit = jj + TILING_SIZE;

        // (ii, jj) is unique, and their c block won't intersect
        for (int kk = 0; kk < n; kk += TILING_SIZE) {
            int k_limit = kk + TILING_SIZE;
            
            for (int i = ii; i < i_limit; i++) {
            for (int k = kk; k < k_limit; k++) {
                float r = a[i*n+k];
                __m512 v_a = _mm512_set1_ps(r);
                
                int j = jj;
                for (; j <= j_limit - 16; j += 16) { // vectorized
                    // c[i*n+j] += r * b[k*n+j];
                    __m512 v_c = _mm512_load_ps(&c[i*n+j]);
                    __m512 v_b = _mm512_load_ps(&b[k*n+j]);
                    
                    // fma
                    v_c = _mm512_fmadd_ps(v_a, v_b, v_c);
                    // store v_res back to c[i*n+j]
                    _mm512_store_ps(&c[i*n+j], v_c);
                } 
                // handle remaining elements
                for (; j < j_limit; j++) {
                    c[i*n+j] += r * b[k*n+j];
                }
            } } 
        } 
    } }
}
#endif

#ifdef DBUG
// ! test
void test_pack_data() {
    std::srand(42);
    int n = 4096;
    std::vector<float> b(n * n);

    for (int i = 0; i < n*n; i++) {
        b[i] = std::rand() / (float)RAND_MAX;
    }

    CacheAlignedVector<float> b_panel(Kc*Nc, 0.0f);

    auto start = std::chrono::high_resolution_clock::now();
    packMatrixB(b.data(), b_panel.data(), 0, 0, n);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time 1: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us" << std::endl;
    float checksum = 0;
    for(int k=0; k<10; ++k) checksum += b_panel[k*100]; 
    std::cout << "Checksum: " << checksum << " (Ignore this, just to prevent optimization)" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    packMatrixB_2(b, b_panel, 0, 0, n);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Time 2: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us" << std::endl;
    checksum = 0;
    for(int k=0; k<10; ++k) checksum += b_panel[k*100]; 
    std::cout << "Checksum: " << checksum << " (Ignore this, just to prevent optimization)" << std::endl;
}

#endif

#ifdef COMPARE
bool verify(const CacheAlignedVector<float>& ref, const std::vector<float>& target, int n) {
    float max_diff = 0.0f;
    for (int i = 0; i < n * n; i++) {
        float diff = std::abs(ref[i] - target[i]);
        if (diff > max_diff) max_diff = diff;
        
        if (diff > 1e-2) { 
            std::cout << "Error at index " << i << ": Ref=" << ref[i] << ", Target=" << target[i] << std::endl;
            return false;
        }
    }
    std::cout << "Verification Passed! Max absolute difference: " << max_diff << std::endl;
    return true;
}
#endif

int main() 
{
#ifdef DBUG
    test_pack_data();
    return 0;
#endif

    std::srand(42);

    // a is 4096x4096 matrix
    int n = 4096;
    std::vector<float> a(n * n);
    std::vector<float> b(n * n);
    std::vector<float> c(n * n, 0.0f);
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
    std::cout << "Matrix Multiplication Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
    
    std::srand(123);
    int validation_idx = std::rand() % (n*n);
    // check the result
    std::cout << "Check the Result at " << validation_idx << " is " << c[validation_idx] << std::endl;

#ifdef COMPARE
    CacheAlignedVector<float> aa = from(a);
    CacheAlignedVector<float> bb = from(b);
    CacheAlignedVector<float> cc(n*n, 0.0f);

    start = std::chrono::high_resolution_clock::now();
    
    mulmat_v6(aa, bb, cc, n);

    end = std::chrono::high_resolution_clock::now();
    // time in seconds
    std::cout << "Matrix Multiplication Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
    
    // check the result
    std::cout << "Check the Result at " << validation_idx << " is " << c[validation_idx] << std::endl;

    // ! DON'T USE SUM TO VERIFY IF TWO MATRICES IS IDENTICAL OR NOT
    // float sum1 = 0;
    // // #pragma omp parallel for reduction(+:sum1) schedule(dynamic)
    // for (int i = 0; i < n; i++) {
    //     for (int j = 0; j < n; j++) {
    //         sum1 += cc[i*n + j];
    //     }
    // }

    // float sum2 = 0;
    // // #pragma omp parallel for reduction(+:sum2) schedule(dynamic)
    // for (int i = 0; i < n; i++) {
    //     for (int j = 0; j < n; j++) {
    //         sum2 += c[i*n + j];
    //     }
    // }

    // std::cout << "the difference of two matmul: " << sum1 - sum2 << std::endl; 

    verify(cc, c, n);

#endif

    return 0;
}


// test
// 0: 425 ms
// 1: 374 ms
// 2: 392 ms
// 3: 390 ms
// 4: 400 ms 
// 5: 448 ms
// 6: 394 ms
// 7: 392 ms
// 8: 391 ms
// 9: 388 ms


