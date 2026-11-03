#include <string.h>

#include "executor.h"
#include "parser.h"
#include "test.h"

static void tiny_mlp_hand_computed(void) {
    /* x = [1,2,3,4] through examples/tiny_mlp.txt, by hand:
     *   x*W1      = [1.9, 0.3, 2.0];   + B1 = [2.0, 0.2, 2.05];   relu: unchanged
     *   h*W2      = [2.395, -0.235];   + B2 = [2.395, -0.135]
     *   softmax   = [1, e^-2.53] / (1 + e^-2.53) = [0.9262184, 0.0737816] */
    char err[512];
    Graph *g = parse_model_file("examples/tiny_mlp.txt", err, sizeof err);
    CHECK(g != NULL);
    int s[2] = {1, 4};
    float xv[4] = {1, 2, 3, 4};
    Tensor *x = tensor_from(2, s, xv);
    ExecStats st;
    Tensor *y = graph_run(g, x, 1, &st, err, sizeof err);
    CHECK(y != NULL);
    const float want[2] = {0.9262184f, 0.0737816f};
    CHECK_ARRAY_NEAR(y->data, want, 2, 1e-6);
    CHECK(st.kernels_run == 6); /* MATMUL ADD RELU MATMUL ADD SOFTMAX */
    tensor_free(x);
    tensor_free(y);
    graph_free(g);
}

static void input_shape_mismatch_is_error(void) {
    char err[512] = "";
    Graph *g = parse_model_file("examples/tiny_mlp.txt", err, sizeof err);
    CHECK(g != NULL);
    int s[2] = {1, 5};
    Tensor *x = tensor_new(2, s);
    Tensor *y = graph_run(g, x, 1, NULL, err, sizeof err);
    CHECK(y == NULL);
    CHECK(strstr(err, "input shape [1x5] does not match INPUT 'x' [1x4]") != NULL);
    tensor_free(x);
    graph_free(g);
}

static Graph *relu_chain(int len, int width) {
    Graph *g = graph_new();
    int s[2] = {1, width};
    int prev = graph_add_input(g, "x", 2, s);
    for (int i = 0; i < len; i++) prev = graph_add_op(g, OP_RELU, "r", &prev, 1);
    g->output = prev;
    char err[256];
    graph_infer_shapes(g, err, sizeof err);
    return g;
}

static void liveness_lowers_peak_memory(void) {
    /* A chain of 4 RELUs over 1000 floats (4000 bytes each).
     *   no reuse: all 4 outputs stay alive          -> peak 16000 bytes
     *   reuse:    at most the input and output of the
     *             current op are alive at once       -> peak  8000 bytes */
    Graph *g = relu_chain(4, 1000);
    int s[2] = {1, 1000};
    Tensor *x = tensor_new(2, s);
    char err[256];
    ExecStats a, b;
    Tensor *y1 = graph_run(g, x, 0, &a, err, sizeof err);
    Tensor *y2 = graph_run(g, x, 1, &b, err, sizeof err);
    CHECK(y1 && y2);
    CHECK(a.peak_bytes == 16000);
    CHECK(b.peak_bytes == 8000);
    CHECK(memcmp(y1->data, y2->data, 4000) == 0);
    tensor_free(x);
    tensor_free(y1);
    tensor_free(y2);
    graph_free(g);
}

static void value_read_twice_is_freed_after_both_reads(void) {
    /* r = relu(x); y = r + r. If r were freed after the first read, ADD would
     * read freed memory. ASan in `make debug` would catch that. */
    Graph *g = graph_new();
    int s[2] = {1, 3};
    int x = graph_add_input(g, "x", 2, s);
    int r = graph_add_op(g, OP_RELU, "r", &x, 1);
    int in[2] = {r, r};
    g->output = graph_add_op(g, OP_ADD, "y", in, 2);
    char err[256];
    CHECK(graph_infer_shapes(g, err, sizeof err) == 0);
    float xv[3] = {-1, 2, 3};
    Tensor *xt = tensor_from(2, s, xv);
    Tensor *y = graph_run(g, xt, 1, NULL, err, sizeof err);
    CHECK(y != NULL);
    const float want[3] = {0, 4, 6};
    CHECK_ARRAY_NEAR(y->data, want, 3, 0.0);
    tensor_free(xt);
    tensor_free(y);
    graph_free(g);
}

static void output_can_be_the_input(void) {
    /* After simplification, a graph may return its input unchanged. The caller must
     * still get its own copy, not a pointer to the input. */
    Graph *g = relu_chain(0, 3);
    int s[2] = {1, 3};
    float xv[3] = {1, 2, 3};
    Tensor *x = tensor_from(2, s, xv);
    char err[256];
    Tensor *y = graph_run(g, x, 1, NULL, err, sizeof err);
    CHECK(y != NULL && y != x);
    CHECK_ARRAY_NEAR(y->data, xv, 3, 0.0);
    tensor_free(x);
    tensor_free(y);
    graph_free(g);
}

static void no_leaks_across_runs(void) {
    size_t before = tensor_mem_live();
    Graph *g = relu_chain(5, 64);
    int s[2] = {1, 64};
    Tensor *x = tensor_new(2, s);
    char err[256];
    for (int i = 0; i < 10; i++) tensor_free(graph_run(g, x, i % 2, NULL, err, sizeof err));
    tensor_free(x);
    graph_free(g);
    CHECK(tensor_mem_live() == before); /* every tensor allocated was freed */
}

void test_exec(void) {
    RUN(tiny_mlp_hand_computed);
    RUN(input_shape_mismatch_is_error);
    RUN(liveness_lowers_peak_memory);
    RUN(value_read_twice_is_freed_after_both_reads);
    RUN(output_can_be_the_input);
    RUN(no_leaks_across_runs);
}
