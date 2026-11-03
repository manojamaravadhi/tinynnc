#include "tensor.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Live/peak byte counters. Every data buffer is allocated and freed in this
 * file, so these counters are exact. */
static size_t g_live_bytes = 0;
static size_t g_peak_bytes = 0;

static void mem_track_alloc(size_t bytes) {
    g_live_bytes += bytes;
    if (g_live_bytes > g_peak_bytes) g_peak_bytes = g_live_bytes;
}

static void mem_track_free(size_t bytes) { g_live_bytes -= bytes; }

size_t tensor_mem_live(void) { return g_live_bytes; }
size_t tensor_mem_peak(void) { return g_peak_bytes; }
void tensor_mem_reset_peak(void) { g_peak_bytes = g_live_bytes; }

size_t shape_numel(int ndim, const int *shape) {
    if (ndim < 1 || ndim > TENSOR_MAX_DIMS || shape == NULL) return 0;
    size_t n = 1;
    for (int i = 0; i < ndim; i++) {
        if (shape[i] <= 0) return 0;
        /* Check before multiplying. A huge shape from a malformed model file
         * must not wrap around to a small size, or we would allocate too
         * little and then write past the end. */
        if (n > SIZE_MAX / sizeof(float) / (size_t)shape[i]) return 0;
        n *= (size_t)shape[i];
    }
    return n;
}

Tensor *tensor_new(int ndim, const int *shape) {
    size_t n = shape_numel(ndim, shape);
    if (n == 0) {
        fprintf(stderr, "tensor_new: invalid shape (ndim=%d)\n", ndim);
        return NULL;
    }
    Tensor *t = malloc(sizeof *t);
    if (t == NULL) {
        fprintf(stderr, "tensor_new: out of memory (struct)\n");
        return NULL;
    }
    /* calloc zero-fills, so a new tensor is always initialized. */
    t->data = calloc(n, sizeof(float));
    if (t->data == NULL) {
        fprintf(stderr, "tensor_new: out of memory (%zu floats)\n", n);
        free(t);
        return NULL;
    }
    t->ndim = ndim;
    for (int i = 0; i < TENSOR_MAX_DIMS; i++) t->shape[i] = (i < ndim) ? shape[i] : 1;
    t->size = n;
    mem_track_alloc(n * sizeof(float));
    return t;
}

Tensor *tensor_from(int ndim, const int *shape, const float *values) {
    Tensor *t = tensor_new(ndim, shape);
    if (t == NULL) return NULL;
    if (values != NULL) memcpy(t->data, values, t->size * sizeof(float));
    return t;
}

Tensor *tensor_clone(const Tensor *t) {
    if (t == NULL) return NULL;
    return tensor_from(t->ndim, t->shape, t->data);
}

void tensor_free(Tensor *t) {
    if (t == NULL) return;
    mem_track_free(t->size * sizeof(float));
    free(t->data);
    free(t);
}

size_t tensor_bytes(const Tensor *t) { return t ? t->size * sizeof(float) : 0; }

int tensor_same_shape(const Tensor *a, const Tensor *b) {
    if (a == NULL || b == NULL || a->ndim != b->ndim) return 0;
    for (int i = 0; i < a->ndim; i++)
        if (a->shape[i] != b->shape[i]) return 0;
    return 1;
}

char *shape_str(int ndim, const int *shape, char *buf, size_t buflen) {
    if (buflen == 0) return buf;
    size_t used = 0;
    buf[0] = '\0';
    int w = snprintf(buf, buflen, "[");
    if (w > 0) used += (size_t)w;
    for (int i = 0; i < ndim && used < buflen; i++) {
        w = snprintf(buf + used, buflen - used, i ? "x%d" : "%d", shape[i]);
        if (w > 0) used += (size_t)w;
    }
    if (used < buflen) snprintf(buf + used, buflen - used, "]");
    return buf;
}

void tensor_print(const Tensor *t, FILE *out) {
    if (t == NULL) {
        fprintf(out, "(null tensor)\n");
        return;
    }
    char sb[64];
    fprintf(out, "tensor %s\n", shape_str(t->ndim, t->shape, sb, sizeof sb));
    /* Print as rows of the last dimension. That is readable for 1D and 2D,
     * which is everything an MLP produces. */
    size_t cols = (size_t)t->shape[t->ndim - 1];
    for (size_t i = 0; i < t->size; i++) {
        fprintf(out, "%s%.6g", (i % cols == 0) ? "  " : " ", t->data[i]);
        if (i % cols == cols - 1) fputc('\n', out);
    }
}
