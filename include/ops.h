#ifndef TINYNNC_OPS_H
#define TINYNNC_OPS_H

/*
 * Compute kernels.
 *
 * Kernels work on raw row-major float arrays with explicit sizes. They do not
 * allocate, free, or check shapes. Shape checking is done once by graph shape
 * inference, before execution, so the hot loops stay simple.
 * Unless a kernel says otherwise, `out` must not alias its inputs.
 *
 * Shapes in the comments use (rows x cols).
 */

#include <stddef.h>

/* C (M x N) = A (M x K) * B (K x N). */
void op_matmul(const float *A, const float *B, float *C, int M, int K, int N);

/* out (M x N) = a (M x N) + b, where b is either (M x N) or a single row
 * (1 x N) that is broadcast to every row. b_rows must be 1 or M. */
void op_add(const float *a, const float *b, float *out, int M, int N, int b_rows);

/* Elementwise ops over n floats. These may run in place (out == in). */
void op_relu(const float *in, float *out, size_t n);
void op_sigmoid(const float *in, float *out, size_t n);

/* Row-wise softmax over a (M x N) matrix. Numerically stable: subtracts the
 * row max before exp(). May run in place. */
void op_softmax(const float *in, float *out, int M, int N);

/* Flatten only changes the shape. The data is copied unchanged. */
void op_copy(const float *in, float *out, size_t n);

/*
 * Fused kernels.
 *
 * op_linear      : C = A*B + bias            (MATMUL + ADD fused)
 * op_linear_relu : C = max(A*B + bias, 0)    (MATMUL + ADD + RELU fused)
 *
 * bias is (bias_rows x N) with bias_rows equal to 1 (broadcast) or M.
 *
 * They are fused because the bias add and the ReLU happen while each output
 * row is still in cache. The unfused path writes the full M x N matmul result
 * to memory, reads it back for ADD, writes it again, and reads it back again
 * for RELU. It also needs two extra intermediate buffers.
 *
 * The floating point operations happen in the same order as the unfused path:
 * the dot product is summed first, and the bias is added after. So the fused
 * output is bit-for-bit identical to MATMUL -> ADD -> RELU, not just within
 * tolerance.
 */
void op_linear(const float *A, const float *B, const float *bias, float *C,
               int M, int K, int N, int bias_rows);
void op_linear_relu(const float *A, const float *B, const float *bias, float *C,
                    int M, int K, int N, int bias_rows);

#endif
