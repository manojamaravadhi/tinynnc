#include <limits.h>
#include <string.h>

#include "tensor.h"
#include "test.h"

static void new_is_zeroed_and_shaped(void) {
    int shape[2] = {2, 3};
    Tensor *t = tensor_new(2, shape);
    CHECK(t != NULL);
    CHECK(t->ndim == 2 && t->shape[0] == 2 && t->shape[1] == 3);
    CHECK(t->shape[2] == 1 && t->shape[3] == 1); /* unused dims are padded with 1 */
    CHECK(t->size == 6);
    for (size_t i = 0; i < t->size; i++) CHECK(t->data[i] == 0.0f);
    tensor_free(t);
}

static void invalid_shapes_return_null(void) {
    /* These go through the same path the parser will use with untrusted
     * model files, so none of them may crash or allocate a wrong size. */
    int zero[2] = {2, 0}, neg[2] = {-1, 3}, ok[1] = {4};
    int huge[4] = {INT_MAX, INT_MAX, INT_MAX, INT_MAX}; /* size_t overflow */
    fprintf(stderr, "    (the 'invalid shape' messages below are expected)\n");
    CHECK(tensor_new(2, zero) == NULL);
    CHECK(tensor_new(2, neg) == NULL);
    CHECK(tensor_new(0, ok) == NULL);
    CHECK(tensor_new(5, ok) == NULL);
    CHECK(tensor_new(4, huge) == NULL);
    CHECK(tensor_new(1, NULL) == NULL);
}

static void from_copies_values(void) {
    float vals[4] = {1, 2, 3, 4};
    int shape[2] = {2, 2};
    Tensor *t = tensor_from(2, shape, vals);
    CHECK(t != NULL);
    vals[0] = 99; /* the tensor owns its own copy */
    CHECK(t->data[0] == 1.0f && t->data[3] == 4.0f);
    tensor_free(t);
}

static void clone_is_independent(void) {
    float vals[3] = {1, 2, 3};
    int shape[1] = {3};
    Tensor *a = tensor_from(1, shape, vals);
    Tensor *b = tensor_clone(a);
    CHECK(a && b && a->data != b->data);
    CHECK(tensor_same_shape(a, b));
    b->data[1] = -5;
    CHECK(a->data[1] == 2.0f);
    tensor_free(a);
    tensor_free(b);
}

static void free_null_is_noop(void) {
    tensor_free(NULL);
    CHECK(tensor_bytes(NULL) == 0);
}

static void same_shape_detects_mismatch(void) {
    int s23[2] = {2, 3}, s32[2] = {3, 2}, s6[1] = {6};
    Tensor *a = tensor_new(2, s23), *b = tensor_new(2, s32), *c = tensor_new(1, s6);
    CHECK(!tensor_same_shape(a, b)); /* same size, different shape */
    CHECK(!tensor_same_shape(a, c)); /* same size, different rank */
    CHECK(!tensor_same_shape(a, NULL));
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

static void memory_accounting(void) {
    size_t base = tensor_mem_live();
    tensor_mem_reset_peak();
    int s[2] = {2, 3};               /* 6 floats = 24 bytes */
    int s2[1] = {10};                /* 10 floats = 40 bytes */
    Tensor *a = tensor_new(2, s);
    CHECK(tensor_mem_live() == base + 24);
    Tensor *b = tensor_new(1, s2);
    CHECK(tensor_mem_live() == base + 64);
    tensor_free(a);
    CHECK(tensor_mem_live() == base + 40);
    CHECK(tensor_mem_peak() == base + 64); /* peak remembers the high-water mark */
    tensor_free(b);
    CHECK(tensor_mem_live() == base);
}

static void shape_string(void) {
    int s[3] = {1, 4, 3};
    char buf[32];
    CHECK(strcmp(shape_str(3, s, buf, sizeof buf), "[1x4x3]") == 0);
    char tiny[4]; /* truncated output must still be NUL-terminated */
    shape_str(3, s, tiny, sizeof tiny);
    CHECK(strlen(tiny) < sizeof tiny);
}

void test_tensor(void) {
    RUN(new_is_zeroed_and_shaped);
    RUN(invalid_shapes_return_null);
    RUN(from_copies_values);
    RUN(clone_is_independent);
    RUN(free_null_is_noop);
    RUN(same_shape_detects_mismatch);
    RUN(memory_accounting);
    RUN(shape_string);
}
