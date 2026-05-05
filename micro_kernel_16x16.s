
# void micro_kernel_16x16(const float* a, const float* b, float* c, int ldc, int K)
# 寄存器约定 (System V ABI): rdi=a, rsi=b, rdx=c, rcx=ldc, r8=K

# micro_kernel_16x16(const float* a, const float* b, float* c, int ldc, int K)
# Windows x64 调用约定寄存器映射:
# rcx: a_panel
# rdx: b_panel
# r8 : c
# r9 : ldc
# [rsp + 40]: K (由于前4个参数占用寄存器，第5个参数在栈上偏移40字节处)

.intel_syntax noprefix
.text
.globl micro_kernel_16x16

micro_kernel_16x16:
    # --- 保存非易失性寄存器 ---
    # 根据 Windows ABI，如果使用了 rsi, rdi, rbx 等需要 push
    # 这里我们只使用易失性寄存器 r10, r11 和 zmm0-zmm17
    
    # 1. 获取第 5 个参数 K
    mov r10, [rsp + 40]     # r10 = K
    test r10, r10           # 如果 K <= 0 直接退出
    jle .L_DONE

    # 2. 处理 ldc (字节偏移 = ldc * 4)
    shl r9, 2               

    # 3. 加载 C 矩阵到 zmm0-zmm15 (16行)
    # 使用 vmovups 处理非对齐地址
    vmovups zmm0,  [r8]
    vmovups zmm1,  [r8 + r9]
    lea r11, [r8 + r9 * 2]  # r11 = c + 2*ldc
    vmovups zmm2,  [r11]
    vmovups zmm3,  [r11 + r9]
    lea r11, [r11 + r9 * 2] # r11 = c + 4*ldc
    vmovups zmm4,  [r11]
    vmovups zmm5,  [r11 + r9]
    lea r11, [r11 + r9 * 2] # r11 = c + 6*ldc
    vmovups zmm6,  [r11]
    vmovups zmm7,  [r11 + r9]
    lea r11, [r11 + r9 * 2] # r11 = c + 8*ldc
    vmovups zmm8,  [r11]
    vmovups zmm9,  [r11 + r9]
    lea r11, [r11 + r9 * 2] # r11 = c + 10*ldc
    vmovups zmm10, [r11]
    vmovups zmm11, [r11 + r9]
    lea r11, [r11 + r9 * 2] # r11 = c + 12*ldc
    vmovups zmm12, [r11]
    vmovups zmm13, [r11 + r9]
    lea r11, [r11 + r9 * 2] # r11 = c + 14*ldc
    vmovups zmm14, [r11]
    vmovups zmm15, [r11 + r9]

    # --- 主循环开始 ---
.L_K_LOOP:
    
    # prefetch B
    prefetcht0 [rdx + 512]

    # prefetch A
    prefetcht0 [rcx + 256]
    # 加载 b_panel 的一行 (16个 float, 64字节)
    # 假设 b_panel 已对齐，使用 vmovaps
    vmovaps zmm16, [rdx]
    
    # 广播 a_panel 的列元素并进行 FMA
    # vfmadd231ps: dest = (src1 * src2) + dest
    vbroadcastss zmm17, [rcx]
    vfmadd231ps zmm0,  zmm16, zmm17
    vbroadcastss zmm17, [rcx + 4]
    vfmadd231ps zmm1,  zmm16, zmm17
    vbroadcastss zmm17, [rcx + 8]
    vfmadd231ps zmm2,  zmm16, zmm17
    vbroadcastss zmm17, [rcx + 12]
    vfmadd231ps zmm3,  zmm16, zmm17
    vbroadcastss zmm17, [rcx + 16]
    vfmadd231ps zmm4,  zmm16, zmm17
    vbroadcastss zmm17, [rcx + 20]
    vfmadd231ps zmm5,  zmm16, zmm17
    vbroadcastss zmm17, [rcx + 24]
    vfmadd231ps zmm6,  zmm16, zmm17
    vbroadcastss zmm17, [rcx + 28]
    vfmadd231ps zmm7,  zmm16, zmm17
    vbroadcastss zmm17, [rcx + 32]
    vfmadd231ps zmm8,  zmm16, zmm17
    vbroadcastss zmm17, [rcx + 36]
    vfmadd231ps zmm9,  zmm16, zmm17
    vbroadcastss zmm17, [rcx + 40]
    vfmadd231ps zmm10, zmm16, zmm17
    vbroadcastss zmm17, [rcx + 44]
    vfmadd231ps zmm11, zmm16, zmm17
    vbroadcastss zmm17, [rcx + 48]
    vfmadd231ps zmm12, zmm16, zmm17
    vbroadcastss zmm17, [rcx + 52]
    vfmadd231ps zmm13, zmm16, zmm17
    vbroadcastss zmm17, [rcx + 56]
    vfmadd231ps zmm14, zmm16, zmm17
    vbroadcastss zmm17, [rcx + 60]
    vfmadd231ps zmm15, zmm16, zmm17

    # 更新指针和计数
    add rcx, 64     # a_panel += 16 * 4
    add rdx, 64     # b_panel += 16 * 4
    dec r10
    jnz .L_K_LOOP

    # 4. 写回结果到 C 矩阵
    vmovups [r8], zmm0
    vmovups [r8 + r9], zmm1
    lea r11, [r8 + r9 * 2]
    vmovups [r11], zmm2
    vmovups [r11 + r9], zmm3
    lea r11, [r11 + r9 * 2]
    vmovups [r11], zmm4
    vmovups [r11 + r9], zmm5
    lea r11, [r11 + r9 * 2]
    vmovups [r11], zmm6
    vmovups [r11 + r9], zmm7
    lea r11, [r11 + r9 * 2]
    vmovups [r11], zmm8
    vmovups [r11 + r9], zmm9
    lea r11, [r11 + r9 * 2]
    vmovups [r11], zmm10
    vmovups [r11 + r9], zmm11
    lea r11, [r11 + r9 * 2]
    vmovups [r11], zmm12
    vmovups [r11 + r9], zmm13
    lea r11, [r11 + r9 * 2]
    vmovups [r11], zmm14
    vmovups [r11 + r9], zmm15

.L_DONE:
    ret


