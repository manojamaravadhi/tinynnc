#define _POSIX_C_SOURCE 200809L /* open_memstream on Linux */
#include <stdlib.h>
#include <string.h>

#include "graph.h"
#include "test.h"

static Tensor *const2(int r, int c, float v) {
    int s[2] = {r, c};
    Tensor *t = tensor_new(2, s);
    if (t)
        for (size_t i = 0; i < t->size; i++) t->data[i] = v;
    return t;
}

/* Position of `id` in a topological order (or -1 if missing). */
static int pos(const int *order, int n, int id) {
    for (int i = 0; i < n; i++)
        if (order[i] == id) return i;
    return -1;
}

/*        x
 *       / \
 *    relu  sigmoid      <- x has two consumers (a diamond, not a chain)
 *       \  /
 *        add            */
static Graph *diamond(int *ids) {
    Graph *g = graph_new();
    int s[2] = {2, 3};
    ids[0] = graph_add_input(g, "x", 2, s);
    ids[1] = graph_add_op(g, OP_RELU, "r", &ids[0], 1);
    ids[2] = graph_add_op(g, OP_SIGMOID, "s", &ids[0], 1);
    int in[2] = {ids[1], ids[2]};
    ids[3] = graph_add_op(g, OP_ADD, "a", in, 2);
    g->output = ids[3];
    return g;
}

static void topo_order_respects_edges(void) {
    int ids[4];
    Graph *g = diamond(ids);
    int order[4], n = 0;
    char err[256];
    CHECK(graph_topo_order(g, order, &n, err, sizeof err) == 0);
    CHECK(n == 4);
    CHECK(pos(order, n, ids[0]) < pos(order, n, ids[1]));
    CHECK(pos(order, n, ids[0]) < pos(order, n, ids[2]));
    CHECK(pos(order, n, ids[1]) < pos(order, n, ids[3]));
    CHECK(pos(order, n, ids[2]) < pos(order, n, ids[3]));
    graph_free(g);
}

static void use_counts_multi_consumer(void) {
    int ids[4], uses[4];
    Graph *g = diamond(ids);
    graph_count_uses(g, uses);
    CHECK(uses[ids[0]] == 2); /* read by relu and sigmoid */
    CHECK(uses[ids[1]] == 1);
    CHECK(uses[ids[3]] == 1); /* the output counts as a consumer */
    graph_free(g);
}

static void cycle_is_detected(void) {
    int ids[4];
    Graph *g = diamond(ids);
    /* The parser cannot create a cycle (names must be defined before use), but
     * a buggy pass could. Make relu read from add, which closes a loop. */
    g->nodes[ids[1]].inputs[0] = ids[3];
    int order[4], n = 0;
    char err[256] = "";
    CHECK(graph_topo_order(g, order, &n, err, sizeof err) == -1);
    CHECK(strstr(err, "cycle") != NULL);
    CHECK(graph_infer_shapes(g, err, sizeof err) == -1);
    graph_free(g);
}

static void self_loop_is_detected(void) {
    int ids[4];
    Graph *g = diamond(ids);
    g->nodes[ids[3]].inputs[0] = ids[3];
    int order[4], n = 0;
    char err[256] = "";
    CHECK(graph_topo_order(g, order, &n, err, sizeof err) == -1);
    graph_free(g);
}

static void dangling_edge_is_error(void) {
    int ids[4];
    Graph *g = diamond(ids);
    g->nodes[ids[1]].dead = 1; /* add still reads relu */
    int order[4], n = 0;
    char err[256] = "";
    CHECK(graph_topo_order(g, order, &n, err, sizeof err) == -1);
    CHECK(strstr(err, "removed") != NULL);
    graph_free(g);
}

static void shapes_inferred_for_mlp_layer(void) {
    Graph *g = graph_new();
    int s[2] = {5, 4};
    int x = graph_add_input(g, "x", 2, s);
    int w = graph_add_const(g, "W", const2(4, 3, 1));
    int b = graph_add_const(g, "b", const2(1, 3, 0));
    int mm_in[2] = {x, w};
    int mm = graph_add_op(g, OP_MATMUL, "mm", mm_in, 2);
    int add_in[2] = {mm, b};
    int add = graph_add_op(g, OP_ADD, "add", add_in, 2);
    int r = graph_add_op(g, OP_SOFTMAX, "p", &add, 1);
    g->output = r;
    char err[256];
    CHECK(graph_infer_shapes(g, err, sizeof err) == 0);
    CHECK(g->nodes[mm].ndim == 2 && g->nodes[mm].shape[0] == 5 && g->nodes[mm].shape[1] == 3);
    CHECK(g->nodes[r].shape[0] == 5 && g->nodes[r].shape[1] == 3);
    GraphStats st;
    graph_stats(g, &st);
    CHECK(st.compute_nodes == 3);
    /* 2*5*4*3 (matmul) + 15 (add) + 4*15 (softmax) = 120 + 15 + 60 */
    CHECK_NEAR(st.flops, 195.0, 0.0);
    CHECK(st.const_bytes == (12 + 3) * sizeof(float));
    graph_free(g);
}

static void matmul_mismatch_message(void) {
    Graph *g = graph_new();
    int s[2] = {1, 4};
    int x = graph_add_input(g, "x", 2, s);
    int w = graph_add_const(g, "W", const2(3, 3, 1)); /* needs 4 rows */
    int in[2] = {x, w};
    g->output = graph_add_op(g, OP_MATMUL, "mm", in, 2);
    char err[256] = "";
    CHECK(graph_infer_shapes(g, err, sizeof err) == -1);
    CHECK(strstr(err, "inner dimensions differ") != NULL);
    CHECK(strstr(err, "4 != 3") != NULL);
    graph_free(g);
}

static void add_mismatch_message(void) {
    Graph *g = graph_new();
    int s[2] = {2, 3};
    int x = graph_add_input(g, "x", 2, s);
    int b = graph_add_const(g, "b", const2(2, 4, 0));
    int in[2] = {x, b};
    g->output = graph_add_op(g, OP_ADD, "a", in, 2);
    char err[256] = "";
    CHECK(graph_infer_shapes(g, err, sizeof err) == -1);
    CHECK(strstr(err, "expected [2x3] or [1x3]") != NULL);
    graph_free(g);
}

static void builder_rejects_bad_args(void) {
    Graph *g = graph_new();
    int s[2] = {2, 2};
    int x = graph_add_input(g, "x", 2, s);
    int bad = 99;
    CHECK(graph_add_op(g, OP_RELU, "r", &bad, 1) == -1);    /* unknown input id */
    int two[2] = {x, x};
    CHECK(graph_add_op(g, OP_RELU, "r", two, 2) == -1);     /* wrong arity */
    CHECK(graph_add_op(g, OP_CONST, "c", NULL, 0) == -1);   /* CONST needs add_const */
    CHECK(graph_add_const(g, "c", NULL) == -1);
    graph_free(g);
}

static void replace_uses_and_clone(void) {
    int ids[4];
    Graph *g = diamond(ids);
    Graph *c = graph_clone(g);
    graph_replace_uses(g, ids[1], ids[0]); /* add now reads x directly */
    CHECK(g->nodes[ids[3]].inputs[0] == ids[0]);
    CHECK(c->nodes[ids[3]].inputs[0] == ids[1]); /* the clone is unaffected */
    graph_replace_uses(g, ids[3], ids[2]);
    CHECK(g->output == ids[2]);
    graph_free(g);
    graph_free(c);
}

static void printers_produce_output(void) {
    int ids[4];
    Graph *g = diamond(ids);
    char err[256];
    CHECK(graph_infer_shapes(g, err, sizeof err) == 0);
    char *buf = NULL;
    size_t len = 0;
    FILE *f = open_memstream(&buf, &len);
    CHECK(f != NULL);
    graph_print(g, f);
    graph_to_dot(g, f, "t");
    fclose(f);
    CHECK(strstr(buf, "ADD(r, s)") != NULL);
    CHECK(strstr(buf, "<- output") != NULL);
    CHECK(strstr(buf, "digraph \"t\"") != NULL);
    CHECK(strstr(buf, "n0 -> n1;") != NULL);
    free(buf);
    graph_free(g);
}

void test_graph(void) {
    RUN(topo_order_respects_edges);
    RUN(use_counts_multi_consumer);
    RUN(cycle_is_detected);
    RUN(self_loop_is_detected);
    RUN(dangling_edge_is_error);
    RUN(shapes_inferred_for_mlp_layer);
    RUN(matmul_mismatch_message);
    RUN(add_mismatch_message);
    RUN(builder_rejects_bad_args);
    RUN(replace_uses_and_clone);
    RUN(printers_produce_output);
}
