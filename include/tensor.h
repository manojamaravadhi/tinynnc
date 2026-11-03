#ifndef TINYNNC_TENSOR_H
#define TINYNNC_TENSOR_H

/*
 * Tensor: a dense, row-major float array with up to 4 dimensions.
 *
 * Ownership rules (the whole project follows these):
 *   - A Tensor returned by tensor_new / tensor_from / tensor_clone OWNS both the
 *     struct and its data buffer. The caller must release it with tensor_free().
 *   - tensor_free(NULL) is a no-op, so cleanup paths can free unconditionally.
 *   - Kernels in ops.h never allocate or free. They only read and write
 *     buffers that the caller owns.
 *
 * Every allocation goes through this module so that we can count live bytes
 * and peak bytes. The executor uses those counters to report peak memory before
 * and after optimization.
 */

#include <stddef.h>
#include <stdio.h>

#define TENSOR_MAX_DIMS 4

typedef struct Tensor {
    float *data;                  /* size floats, row-major */
    int ndim;                     /* 1..TENSOR_MAX_DIMS */
    int shape[TENSOR_MAX_DIMS];   /* unused trailing dims are set to 1 */
    size_t size;                  /* product of shape[0..ndim-1] */
} Tensor;

/* Number of elements for a shape. Returns 0 if the shape is invalid
 * (ndim out of range, a dimension <= 0, or size_t overflow). */
size_t shape_numel(int ndim, const int *shape);

/* Allocate a zero-filled tensor. Returns NULL (and prints why to stderr) on an
 * invalid shape or out of memory. */
Tensor *tensor_new(int ndim, const int *shape);

/* Allocate a tensor and copy `values` (which must hold numel floats) into it. */
Tensor *tensor_from(int ndim, const int *shape, const float *values);

/* Deep copy. The clone shares nothing with the source. */
Tensor *tensor_clone(const Tensor *t);

void tensor_free(Tensor *t);

/* Bytes used by the data buffer (the small struct itself is not counted). */
size_t tensor_bytes(const Tensor *t);

/* 1 if both tensors have identical ndim and shape, else 0. */
int tensor_same_shape(const Tensor *a, const Tensor *b);

/* Writes a shape string like "[2x3]" into buf. Returns buf. */
char *shape_str(int ndim, const int *shape, char *buf, size_t buflen);

void tensor_print(const Tensor *t, FILE *out);

/* Fills t with deterministic pseudo-random values in [-scale, scale), using a
 * simple LCG. The same seed gives the same values on every platform, so tests and
 * benchmarks are reproducible. *seed is advanced. */
void tensor_fill_random(Tensor *t, unsigned *seed, float scale);

/* Memory accounting for tensor data buffers, across the whole process.
 * Not thread-safe. Allocation only happens on the main thread. */
size_t tensor_mem_live(void);
size_t tensor_mem_peak(void);
void tensor_mem_reset_peak(void);   /* peak := current live */

#endif
