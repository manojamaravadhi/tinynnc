#include "ops.h"

#include <math.h>
#include <string.h>

/*
 * Computes one output row: Crow (1 x N) = Arow (1 x K) * B (K x N).
 *
 * The loop order is i-k-j instead of the textbook i-j-k. The inner loop then
 * walks both B[k, :] and Crow[:] contiguously, which is cache-friendly and easy
 * for the compiler to vectorize. With i-j-k, the inner loop walks down a
 * column of B with stride N, which misses the cache on large matrices.
 *
 * Both the plain and the fused kernels share this routine, so they
 * accumulate in exactly the same order.
 */
static void matmul_row(const float *Arow, const float *B, float *Crow, int K, int N) {
    for (int j = 0; j < N; j++) Crow[j] = 0.0f;
    for (int k = 0; k < K; k++) {
        const float a = Arow[k];
        const float *Brow = B + (size_t)k * N;
        for (int j = 0; j < N; j++) Crow[j] += a * Brow[j];
    }
}

void op_matmul(const float *A, const float *B, float *C, int M, int K, int N) {
    for (int i = 0; i < M; i++)
        matmul_row(A + (size_t)i * K, B, C + (size_t)i * N, K, N);
}

void op_add(const float *a, const float *b, float *out, int M, int N, int b_rows) {
    for (int i = 0; i < M; i++) {
        /* With broadcasting (b_rows == 1), every row reads row 0 of b. */
        const float *brow = b + (size_t)(b_rows == 1 ? 0 : i) * N;
        const float *arow = a + (size_t)i * N;
        float *orow = out + (size_t)i * N;
        for (int j = 0; j < N; j++) orow[j] = arow[j] + brow[j];
    }
}

void op_relu(const float *in, float *out, size_t n) {
    for (size_t i = 0; i < n; i++) out[i] = in[i] > 0.0f ? in[i] : 0.0f;
}

void op_sigmoid(const float *in, float *out, size_t n) {
    for (size_t i = 0; i < n; i++) {
        const float x = in[i];
        /* Pick the form whose exp() argument is never positive. Then exp()
         * cannot overflow to inf, even for x = -1000. */
        if (x >= 0.0f) {
            out[i] = 1.0f / (1.0f + expf(-x));
        } else {
            const float e = expf(x);
            out[i] = e / (1.0f + e);
        }
    }
}

void op_softmax(const float *in, float *out, int M, int N) {
    for (int i = 0; i < M; i++) {
        const float *x = in + (size_t)i * N;
        float *y = out + (size_t)i * N;
        /* softmax(x) == softmax(x - max). Subtracting the max keeps every
         * exponent <= 0, so expf() never overflows (e.g. logits of 1000). */
        float mx = x[0];
        for (int j = 1; j < N; j++) mx = x[j] > mx ? x[j] : mx;
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            y[j] = expf(x[j] - mx);
            sum += y[j];
        }
        const float inv = 1.0f / sum; /* sum >= 1 because the max term is exp(0) */
        for (int j = 0; j < N; j++) y[j] *= inv;
    }
}

void op_copy(const float *in, float *out, size_t n) {
    if (in != out) memcpy(out, in, n * sizeof(float));
}

/* Fused body shared by LINEAR and LINEAR_RELU. `relu` is a compile-time
 * constant at each call site, so the compiler can remove the branch. */
static void linear_impl(const float *A, const float *B, const float *bias, float *C,
                        int M, int K, int N, int bias_rows, int relu) {
    for (int i = 0; i < M; i++) {
        float *Crow = C + (size_t)i * N;
        matmul_row(A + (size_t)i * K, B, Crow, K, N);
        /* Epilogue: Crow was just written, so it is still in L1 cache.
         * The bias add and the ReLU happen here, with no extra pass over memory
         * and no intermediate tensor. */
        const float *brow = bias + (size_t)(bias_rows == 1 ? 0 : i) * N;
        for (int j = 0; j < N; j++) {
            const float v = Crow[j] + brow[j];
            /* Same test as op_relu (x > 0 keeps x). Writing it as `v <= 0`
             * would give a different result for NaN. */
            Crow[j] = (relu && !(v > 0.0f)) ? 0.0f : v;
        }
    }
}

void op_linear(const float *A, const float *B, const float *bias, float *C,
               int M, int K, int N, int bias_rows) {
    linear_impl(A, B, bias, C, M, K, N, bias_rows, 0);
}

void op_linear_relu(const float *A, const float *B, const float *bias, float *C,
                    int M, int K, int N, int bias_rows) {
    linear_impl(A, B, bias, C, M, K, N, bias_rows, 1);
}
