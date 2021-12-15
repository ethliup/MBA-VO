// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2015 Google Inc. All rights reserved.
// http://ceres-solver.org/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: sameeragarwal@google.com (Sameer Agarwal)
//
// Simple blas functions for use in the Schur Eliminator. These are
// fairly basic implementations which already yield a significant
// speedup in the eliminator performance.

#ifndef SLAM_CORE_SMALL_BLAS_H_
#define SLAM_CORE_SMALL_BLAS_H_

#include "SmallBlasGeneric.h"

namespace SLAM
{
    namespace Core
    {
// The following three macros are used to share code and reduce
// template junk across the various GEMM variants.
#define CERES_GEMM_BEGIN(name)                                       \
    template <typename T1, typename T2, typename T3, int kOperation> \
    __FORCEINLINE__ __CPU_AND_CUDA_CODE__ void                       \
    name(const T1 *A,                                                \
         const int num_row_a,                                        \
         const int num_col_a,                                        \
         const T2 *B,                                                \
         const int num_row_b,                                        \
         const int num_col_b,                                        \
         T3 *C,                                                      \
         const int start_row_c,                                      \
         const int start_col_c,                                      \
         const int row_stride_c,                                     \
         const int col_stride_c)

#define CERES_GEMM_NAIVE_HEADER      \
    const int NUM_ROW_A = num_row_a; \
    const int NUM_COL_A = num_col_a; \
    const int NUM_ROW_B = num_row_b; \
    const int NUM_COL_B = num_col_b;

#define CERES_CALL_GEMM(name)    \
    name<kOperation>(            \
        A, num_row_a, num_col_a, \
        B, num_row_b, num_col_b, \
        C, start_row_c, start_col_c, row_stride_c, col_stride_c);

#define CERES_GEMM_STORE_SINGLE(p, index, value) \
    if (kOperation > 0)                          \
    {                                            \
        p[index] += value;                       \
    }                                            \
    else if (kOperation < 0)                     \
    {                                            \
        p[index] -= value;                       \
    }                                            \
    else                                         \
    {                                            \
        p[index] = value;                        \
    }

#define CERES_GEMM_STORE_PAIR(p, index, v1, v2) \
    if (kOperation > 0)                         \
    {                                           \
        p[index] += v1;                         \
        p[index + 1] += v2;                     \
    }                                           \
    else if (kOperation < 0)                    \
    {                                           \
        p[index] -= v1;                         \
        p[index + 1] -= v2;                     \
    }                                           \
    else                                        \
    {                                           \
        p[index] = v1;                          \
        p[index + 1] = v2;                      \
    }

        // For the matrix-matrix functions below, there are three variants for
        // each functionality. Foo, FooNaive and FooEigen. Foo is the one to
        // be called by the user. FooNaive is a basic loop based
        // implementation and FooEigen uses Eigen's implementation. Foo
        // chooses between FooNaive and FooEigen depending on how many of the
        // template arguments are fixed at compile time. Currently, FooEigen
        // is called if all matrix dimensions are compile time
        // constants. FooNaive is called otherwise. This leads to the best
        // performance currently.
        //
        // The MatrixMatrixMultiply variants compute:
        //
        //   C op A * B;
        //
        // The MatrixTransposeMatrixMultiply variants compute:
        //
        //   C op A' * B
        //
        // where op can be +=, -=, or =.
        //
        // The template parameters (kRowA, kColA, kRowB, kColB) allow
        // specialization of the loop at compile time. If this information is
        // not available, then Eigen::Dynamic should be used as the template
        // argument.
        //
        //   kOperation =  1  -> C += A * B
        //   kOperation = -1  -> C -= A * B
        //   kOperation =  0  -> C  = A * B
        //
        // The functions can write into matrices C which are larger than the
        // matrix A * B. This is done by specifying the true size of C via
        // row_stride_c and col_stride_c, and then indicating where A * B
        // should be written into by start_row_c and start_col_c.
        //
        // Graphically if row_stride_c = 10, col_stride_c = 12, start_row_c =
        // 4 and start_col_c = 5, then if A = 3x2 and B = 2x4, we get
        //
        //   ------------
        //   ------------
        //   ------------
        //   ------------
        //   -----xxxx---
        //   -----xxxx---
        //   -----xxxx---
        //   ------------
        //   ------------
        //   ------------
        //
        CERES_GEMM_BEGIN(MatrixMatrixMultiply)
        {
            CERES_GEMM_NAIVE_HEADER

            const int NUM_ROW_C = NUM_ROW_A;
            const int NUM_COL_C = NUM_COL_B;
            const int span = 4;

            // Calculate the remainder part first.

            // Process the last odd column if present.
            if (NUM_COL_C & 1)
            {
                int col = NUM_COL_C - 1;
                const T1 *pa = &A[0];
                for (int row = 0; row < NUM_ROW_C; ++row, pa += NUM_COL_A)
                {
                    const T2 *pb = &B[col];
                    T3 tmp = 0.0;
                    for (int k = 0; k < NUM_COL_A; ++k, pb += NUM_COL_B)
                    {
                        tmp += pa[k] * pb[0];
                    }

                    const int index = (row + start_row_c) * col_stride_c + start_col_c + col;
                    CERES_GEMM_STORE_SINGLE(C, index, tmp);
                }

                // Return directly for efficiency of extremely small matrix multiply.
                if (NUM_COL_C == 1)
                {
                    return;
                }
            }

            // Process the couple columns in remainder if present.
            if (NUM_COL_C & 2)
            {
                int col = NUM_COL_C & (int)(~(span - 1));
                const T1 *pa = &A[0];
                for (int row = 0; row < NUM_ROW_C; ++row, pa += NUM_COL_A)
                {
                    const T2 *pb = &B[col];
                    T3 tmp1 = 0.0, tmp2 = 0.0;
                    for (int k = 0; k < NUM_COL_A; ++k, pb += NUM_COL_B)
                    {
                        T1 av = pa[k];
                        tmp1 += av * pb[0];
                        tmp2 += av * pb[1];
                    }

                    const int index = (row + start_row_c) * col_stride_c + start_col_c + col;
                    CERES_GEMM_STORE_PAIR(C, index, tmp1, tmp2);
                }

                // Return directly for efficiency of extremely small matrix multiply.
                if (NUM_COL_C < span)
                {
                    return;
                }
            }

            // Calculate the main part with multiples of 4.
            int col_m = NUM_COL_C & (int)(~(span - 1));
            for (int col = 0; col < col_m; col += span)
            {
                for (int row = 0; row < NUM_ROW_C; ++row)
                {
                    const int index = (row + start_row_c) * col_stride_c + start_col_c + col;
                    MMM_mat1x4(NUM_COL_A, &A[row * NUM_COL_A],
                               &B[col], NUM_COL_B, &C[index], kOperation);
                }
            }
        }

        CERES_GEMM_BEGIN(MatrixTransposeMatrixMultiply)
        {
            CERES_GEMM_NAIVE_HEADER

            const int NUM_ROW_C = NUM_COL_A;
            const int NUM_COL_C = NUM_COL_B;
            const int span = 4;

            // Process the remainder part first.

            // Process the last odd column if present.
            if (NUM_COL_C & 1)
            {
                int col = NUM_COL_C - 1;
                for (int row = 0; row < NUM_ROW_C; ++row)
                {
                    const T1 *pa = &A[row];
                    const T2 *pb = &B[col];
                    T3 tmp = 0.0;
                    for (int k = 0; k < NUM_ROW_A; ++k)
                    {
                        tmp += pa[0] * pb[0];
                        pa += NUM_COL_A;
                        pb += NUM_COL_B;
                    }

                    const int index = (row + start_row_c) * col_stride_c + start_col_c + col;
                    CERES_GEMM_STORE_SINGLE(C, index, tmp);
                }

                // Return directly for efficiency of extremely small matrix multiply.
                if (NUM_COL_C == 1)
                {
                    return;
                }
            }

            // Process the couple columns in remainder if present.
            if (NUM_COL_C & 2)
            {
                int col = NUM_COL_C & (int)(~(span - 1));
                for (int row = 0; row < NUM_ROW_C; ++row)
                {
                    const T1 *pa = &A[row];
                    const T2 *pb = &B[col];
                    T3 tmp1 = 0.0, tmp2 = 0.0;
                    for (int k = 0; k < NUM_ROW_A; ++k)
                    {
                        T1 av = *pa;
                        tmp1 += av * pb[0];
                        tmp2 += av * pb[1];
                        pa += NUM_COL_A;
                        pb += NUM_COL_B;
                    }

                    const int index = (row + start_row_c) * col_stride_c + start_col_c + col;
                    CERES_GEMM_STORE_PAIR(C, index, tmp1, tmp2);
                }

                // Return directly for efficiency of extremely small matrix multiply.
                if (NUM_COL_C < span)
                {
                    return;
                }
            }

            // Process the main part with multiples of 4.
            int col_m = NUM_COL_C & (int)(~(span - 1));
            for (int col = 0; col < col_m; col += span)
            {
                for (int row = 0; row < NUM_ROW_C; ++row)
                {
                    const int index = (row + start_row_c) * col_stride_c + start_col_c + col;
                    MTM_mat1x4(NUM_ROW_A, &A[row], NUM_COL_A,
                               &B[col], NUM_COL_B, &C[index], kOperation);
                }
            }
        }

        // Matrix-Vector multiplication
        //
        // c op A * b;
        //
        // where op can be +=, -=, or =.
        //
        // The template parameters (kRowA, kColA) allow specialization of the
        // loop at compile time. If this information is not available, then
        // Eigen::Dynamic should be used as the template argument.
        //
        // kOperation =  1  -> c += A * b
        // kOperation = -1  -> c -= A * b
        // kOperation =  0  -> c  = A * b
        template <typename T1, typename T2, typename T3, int kOperation>
        __FORCEINLINE__ __CPU_AND_CUDA_CODE__ void
        MatrixVectorMultiply(const T1 *A,
                             const int num_row_a,
                             const int num_col_a,
                             const T2 *b,
                             T3 *c)
        {
            const int NUM_ROW_A = num_row_a;
            const int NUM_COL_A = num_col_a;
            const int span = 4;

            // Calculate the remainder part first.

            // Process the last odd row if present.
            if (NUM_ROW_A & 1)
            {
                int row = NUM_ROW_A - 1;
                const T1 *pa = &A[row * NUM_COL_A];
                const T2 *pb = &b[0];
                T3 tmp = 0.0;
                for (int col = 0; col < NUM_COL_A; ++col)
                {
                    tmp += (*pa++) * (*pb++);
                }
                CERES_GEMM_STORE_SINGLE(c, row, tmp);

                // Return directly for efficiency of extremely small matrix multiply.
                if (NUM_ROW_A == 1)
                {
                    return;
                }
            }

            // Process the couple rows in remainder if present.
            if (NUM_ROW_A & 2)
            {
                int row = NUM_ROW_A & (int)(~(span - 1));
                const T1 *pa1 = &A[row * NUM_COL_A];
                const T1 *pa2 = pa1 + NUM_COL_A;
                const T2 *pb = &b[0];
                T3 tmp1 = 0.0, tmp2 = 0.0;
                for (int col = 0; col < NUM_COL_A; ++col)
                {
                    T2 bv = *pb++;
                    tmp1 += *(pa1++) * bv;
                    tmp2 += *(pa2++) * bv;
                }
                CERES_GEMM_STORE_PAIR(c, row, tmp1, tmp2);

                // Return directly for efficiency of extremely small matrix multiply.
                if (NUM_ROW_A < span)
                {
                    return;
                }
            }

            // Calculate the main part with multiples of 4.
            int row_m = NUM_ROW_A & (int)(~(span - 1));
            for (int row = 0; row < row_m; row += span)
            {
                MVM_mat4x1(NUM_COL_A, &A[row * NUM_COL_A], NUM_COL_A,
                           &b[0], &c[row], kOperation);
            }
        }

        // Similar to MatrixVectorMultiply, except that A is transposed, i.e.,
        //
        // c op A' * b;
        template <typename T1, typename T2, typename T3, int kOperation>
        __FORCEINLINE__ __CPU_AND_CUDA_CODE__ void
        MatrixTransposeVectorMultiply(const T1 *A,
                                      const int num_row_a,
                                      const int num_col_a,
                                      const T2 *b,
                                      T3 *c)
        {
            const int NUM_ROW_A = num_row_a;
            const int NUM_COL_A = num_col_a;
            const int span = 4;

            // Calculate the remainder part first.

            // Process the last odd column if present.
            if (NUM_COL_A & 1)
            {
                int row = NUM_COL_A - 1;
                const T1 *pa = &A[row];
                const T2 *pb = &b[0];
                T3 tmp = 0.0;
                for (int col = 0; col < NUM_ROW_A; ++col)
                {
                    tmp += *pa * (*pb++);
                    pa += NUM_COL_A;
                }
                CERES_GEMM_STORE_SINGLE(c, row, tmp);

                // Return directly for efficiency of extremely small matrix multiply.
                if (NUM_COL_A == 1)
                {
                    return;
                }
            }

            // Process the couple columns in remainder if present.
            if (NUM_COL_A & 2)
            {
                int row = NUM_COL_A & (int)(~(span - 1));
                const T1 *pa = &A[row];
                const T2 *pb = &b[0];
                T3 tmp1 = 0.0, tmp2 = 0.0;
                for (int col = 0; col < NUM_ROW_A; ++col)
                {
                    T2 bv = *pb++;
                    tmp1 += *(pa)*bv;
                    tmp2 += *(pa + 1) * bv;
                    pa += NUM_COL_A;
                }
                CERES_GEMM_STORE_PAIR(c, row, tmp1, tmp2);

                // Return directly for efficiency of extremely small matrix multiply.
                if (NUM_COL_A < span)
                {
                    return;
                }
            }

            // Calculate the main part with multiples of 4.
            int row_m = NUM_COL_A & (int)(~(span - 1));
            for (int row = 0; row < row_m; row += span)
            {
                MTV_mat4x1(NUM_ROW_A, &A[row], NUM_COL_A,
                           &b[0], &c[row], kOperation);
            }
        }

#undef CERES_GEMM_BEGIN
#undef CERES_GEMM_EIGEN_HEADER
#undef CERES_GEMM_NAIVE_HEADER
#undef CERES_CALL_GEMM
#undef CERES_GEMM_STORE_SINGLE
#undef CERES_GEMM_STORE_PAIR

        template <typename T1, typename T2, typename T3>
        __FORCEINLINE__ __CPU_AND_CUDA_CODE__ void
        MatrixDiagonalAddition(T1 *A, int num_row_a, int num_col_a, T2 *D, T3 *C)
        {
            const int N = num_row_a > num_col_a ? num_col_a : num_row_a;
            for (int i = 0; i < N; i++)
            {
                int idx = i * num_col_a + i;
                C[idx] = A[idx] + D[i];
            }
        }
    } // namespace Core
} // namespace SLAM

#endif // SLAM_CORE_SMALL_BLAS_H_
