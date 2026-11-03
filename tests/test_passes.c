#include <math.h>
#include <string.h>

#include "executor.h"
#include "parser.h"
#include "passes.h"
#include "test.h"

static Graph *parse(const char *text) {
    char err[512];
    Graph *g = parse_model_string(text, "t", err, sizeof err);
    if (g == NULL) fprintf(stderr, "    parse error: %s\n", err);
    return g;
}

static int live_compute(const Graph *g) {
    GraphStats s;
    graph_stats(g, &s);
    return s.compute_nodes;
}

static void dce_removes_unreachable(void) {
    Graph *g = parse("INPUT x 1 2\nRELU r x\nSIGMOID dead1 x\nRELU dead2 dead1\nOUTPUT r\n");
    CHECK(g != NULL);
    CHECK(pass_dce(g) == 2);
    CHECK(graph_find(g, "dead1") < 0 && graph_find(g, "dead2") < 0);
    CHECK(pass_dce(g) == 0); /* idempotent */
    graph_free(g);
}

static void fold_folds_constant_chain(void) {
    /* c = a + b and d = relu(c) only touch constants, so both fold. out = x + d does not. */
    Graph *g = parse("INPUT x 1 2\nCONST a 1 2 1 -5\nCONST b 1 2 2 1\n"
                     "ADD c a b\nRELU d c\nADD out x d\nOUTPUT out\n");
    CHECK(g != NULL);
    CHECK(pass_constant_fold(g) == 2);
    const Node *d = &g->nodes[graph_find(g, "d")];
    CHECK(d->op == OP_CONST && d->ninputs == 0);
    CHECK(d->value->data[0] == 3.0f && d->value->data[1] == 0.0f); /* relu([3,-4]) */
    CHECK(g->nodes[graph_find(g, "out")].op == OP_ADD);
    CHECK(pass_dce(g) == 3); /* a, b and c are now unused */
    graph_free(g);
}

static void fold_whole_graph_to_constant(void) {
    /* The output does not depend on x at all, so the whole thing folds and running
     * the graph just returns (a copy of) the constant. */
    Graph *g = parse("INPUT x 1 2\nCONST a 1 2 1 2\nRELU r a\nOUTPUT r\n");
    CHECK(g != NULL);
    char err[256];
    CHECK(run_passes(g, PASS_ALL, NULL, NULL, err, sizeof err) == 0);
    CHECK(live_compute(g) == 0);
    int s[2] = {1, 2};
    Tensor *x = tensor_new(2, s);
    Tensor *y = graph_run(g, x, 1, NULL, err, sizeof err);
    CHECK(y && y->data[0] == 1.0f && y->data[1] == 2.0f);
    tensor_free(x);
    tensor_free(y);
    graph_free(g);
}

static void simplify_patterns(void) {
    Graph *g = parse("INPUT x 2 2\n"
                     "RELU r1 x\nRELU r2 r1\nRELU r3 r2\n"         /* 2 rewrites */
                     "CONST Z 1 2 0 -0\nADD z r3 Z\n"              /* + zero (incl. -0) */
                     "CONST I 2 2 1 0 0 1\nMATMUL i z I\n"         /* identity, right side */
                     "MATMUL i2 I i\n"                             /* identity, left side */
                     "FLATTEN f i2\nOUTPUT f\n");
    CHECK(g != NULL);
    CHECK(pass_simplify(g) == 6);
    CHECK(g->output == graph_find(g, "r1")); /* everything collapsed to RELU(x) */
    CHECK(pass_simplify(g) == 0);             /* nothing left to do: fixed point */
    graph_free(g);
}

static void simplify_keeps_relu_after_sigmoid(void) {
    /* RELU(SIGMOID(v)) is not removed, because RELU(NaN)=0 but SIGMOID(NaN)=NaN. */
    Graph *g = parse("INPUT x 1 2\nSIGMOID s x\nRELU r s\nOUTPUT r\n");
    CHECK(g != NULL);
    CHECK(pass_simplify(g) == 0);
    graph_free(g);
}

static void simplify_ignores_non_identity(void) {
    Graph *g = parse("INPUT x 1 2\nCONST P 2 2 0 1 1 0\nMATMUL m x P\n"   /* permutation */
                     "CONST Z 1 2 0 0.001\nADD a m Z\nOUTPUT a\n");          /* not zero */
    CHECK(g != NULL);
    CHECK(pass_simplify(g) == 0);
    graph_free(g);
}

static void cse_merges_and_cascades(void) {
    Graph *g = parse("INPUT x 1 2\nCONST W 2 2 1 2 3 4\nCONST b 1 2 1 1\n"
                     "MATMUL m1 x W\nMATMUL m2 x W\n"
                     "ADD a1 m1 b\nADD a2 m2 b\n"        /* identical once m2 -> m1 */
                     "ADD out a1 a2\nOUTPUT out\n");
    CHECK(g != NULL);
    CHECK(pass_cse(g) == 2);
    const Node *out = &g->nodes[graph_find(g, "out")];
    CHECK(out->inputs[0] == out->inputs[1]); /* now ADD(a1, a1) */
    graph_free(g);
}

static void cse_dedupes_identical_constants(void) {
    Graph *g = parse("INPUT x 1 2\nCONST b1 1 2 1 2\nCONST b2 1 2 1 2\nCONST b3 1 2 1 3\n"
                     "ADD a x b1\nADD c a b2\nADD d c b3\nOUTPUT d\n");
    CHECK(g != NULL);
    CHECK(pass_cse(g) == 1); /* b2 == b1 bit for bit; b3 differs */
    graph_free(g);
}

static void cse_respects_operand_order(void) {
    Graph *g = parse("INPUT x 1 2\nRELU r x\nADD a x r\nADD b r x\nADD o a b\nOUTPUT o\n");
    CHECK(g != NULL);
    CHECK(pass_cse(g) == 0);
    graph_free(g);
}

static void pass_spec_parsing(void) {
    unsigned m;
    char err[256] = "";
    CHECK(passes_parse("none", &m, err, sizeof err) == 0 && m == 0);
    CHECK(passes_parse("all", &m, err, sizeof err) == 0 && m == PASS_ALL);
    CHECK(passes_parse("fold,dce", &m, err, sizeof err) == 0 && m == (PASS_FOLD | PASS_DCE));
    CHECK(passes_parse("fold,bogus", &m, err, sizeof err) == -1);
    CHECK(strstr(err, "unknown pass 'bogus'") != NULL);
}

static void redundant_example_end_to_end(void) {
    char err[512];
    Graph *g = parse_model_file("examples/redundant.txt", err, sizeof err);
    CHECK(g != NULL);
    PassReport r;
    CHECK(run_passes(g, PASS_ALL & ~PASS_FUSE, &r, NULL, err, sizeof err) == 0);
    CHECK(r.fold == 1 && r.simplify == 4 && r.cse == 1);
    CHECK(live_compute(g) == 6); /* MATMUL ADD RELU MATMUL ADD SOFTMAX */
    graph_free(g);
}

/* ---------------------------------------------------------------------------- */
/* Equivalence on random graphs                                                 */
/* ---------------------------------------------------------------------------- */

static unsigned g_rng = 2024u;
static unsigned rnd(unsigned n) {
    g_rng = g_rng * 1103515245u + 12345u;
    return (g_rng >> 8) % n;
}

static int add_const(Graph *g, int r, int c, float scale, int kind) {
    int s[2] = {r, c};
    Tensor *t = tensor_new(2, s);
    if (t == NULL) return -1;
    if (kind == 1) { /* identity */
        for (int i = 0; i < r && i < c; i++) t->data[(size_t)i * c + i] = 1.0f;
    } else if (kind == 0) {
        tensor_fill_random(t, &g_rng, scale);
    } /* kind 2: zeros */
    return graph_add_const(g, "c", t);
}

static int op1(Graph *g, OpType op, int a) { return graph_add_op(g, op, "v", &a, 1); }
static int op2(Graph *g, OpType op, int a, int b) {
    int in[2] = {a, b};
    return graph_add_op(g, op, "v", in, 2);
}

/* A random MLP-shaped DAG that contains every pattern the passes look for:
 * redundant activations, identity/zero constants, constant subexpressions,
 * duplicate ops, dead branches, skip connections (a value with several
 * consumers), and MATMUL results shared between two consumers. */
static Graph *random_graph(int *batch_out, int *width_out) {
    Graph *g = graph_new();
    int B = 1 + (int)rnd(4), w = 1 + (int)rnd(12);
    int s[2] = {B, w};
    int cur = graph_add_input(g, "x", 2, s);
    int skip = -1, skip_w = -1;
    int layers = 1 + (int)rnd(4);
    for (int l = 0; l < layers; l++) {
        int n = 1 + (int)rnd(16);
        int W = add_const(g, w, n, 1.0f / sqrtf((float)w), 0);
        int bias = add_const(g, rnd(2) ? 1 : B, n, 0.2f, 0);
        if (rnd(4) == 0) /* bias = c1 + c2 -> fold */
            bias = op2(g, OP_ADD, bias, add_const(g, 1, n, 0.1f, 0));
        int m = op2(g, OP_MATMUL, cur, W);
        if (rnd(4) == 0) op1(g, OP_SIGMOID, op2(g, OP_MATMUL, cur, W)); /* dup + dead */
        int a = rnd(5) == 0 ? op2(g, OP_ADD, add_const(g, B, n, 0.0f, 2), m) /* 0 + m */
                            : op2(g, OP_ADD, m, bias);
        if (rnd(5) == 0) a = op2(g, OP_ADD, a, bias);
        unsigned act = rnd(3);
        if (act == 0) {
            a = op1(g, OP_RELU, a);
            if (rnd(3) == 0) a = op1(g, OP_RELU, a);
        } else if (act == 1) {
            a = op1(g, OP_SIGMOID, a);
        }
        if (rnd(5) == 0) a = op2(g, OP_MATMUL, a, add_const(g, n, n, 0, 1)); /* identity */
        if (rnd(6) == 0) a = op2(g, OP_ADD, a, op1(g, OP_SIGMOID, m));       /* m shared */
        if (skip >= 0 && skip_w == n && rnd(2)) a = op2(g, OP_ADD, a, skip); /* skip conn. */
        if (rnd(3) == 0) {
            skip = a;
            skip_w = n;
        }
        cur = a;
        w = n;
    }
    if (rnd(3) == 0) cur = op1(g, OP_FLATTEN, cur);
    if (rnd(2)) cur = op1(g, OP_SOFTMAX, cur);
    g->output = cur;
    char err[256];
    if (graph_infer_shapes(g, err, sizeof err) != 0) {
        fprintf(stderr, "    generator bug: %s\n", err);
        graph_free(g);
        return NULL;
    }
    *batch_out = B;
    *width_out = g->nodes[graph_input_id(g)].shape[1];
    return g;
}

/* Runs `iters` random graphs: the unoptimized output vs the output with each
 * pass mask in `masks`. Returns the largest |difference| seen (or INFINITY on error). */
static double equivalence_sweep(const unsigned *masks, int nmasks, int iters, int *nodes_removed) {
    double worst = 0;
    *nodes_removed = 0;
    for (int it = 0; it < iters; it++) {
        int B, w;
        Graph *g = random_graph(&B, &w);
        if (g == NULL) return INFINITY;
        int s[2] = {B, w};
        Tensor *x = tensor_new(2, s);
        tensor_fill_random(x, &g_rng, 2.0f);
        char err[256];
        Tensor *ref = graph_run(g, x, 1, NULL, err, sizeof err);
        for (int m = 0; m < nmasks && ref; m++) {
            Graph *o = graph_clone(g);
            Tensor *y = NULL;
            if (o && run_passes(o, masks[m], NULL, NULL, err, sizeof err) == 0)
                y = graph_run(o, x, 1, NULL, err, sizeof err);
            if (y == NULL || !tensor_same_shape(y, ref)) {
                fprintf(stderr, "    graph %d mask 0x%x failed: %s\n", it, masks[m], err);
                worst = INFINITY;
            } else {
                for (size_t i = 0; i < y->size; i++) {
                    double d = fabs((double)y->data[i] - (double)ref->data[i]);
                    if (!(d <= worst)) worst = d;
                }
                *nodes_removed += live_compute(g) - live_compute(o);
            }
            tensor_free(y);
            graph_free(o);
        }
        tensor_free(ref);
        tensor_free(x);
        graph_free(g);
    }
    return worst;
}

static void random_graphs_cleanup_passes_equivalent(void) {
    const unsigned masks[] = {PASS_FOLD, PASS_SIMPLIFY, PASS_CSE, PASS_DCE,
                              PASS_ALL & ~PASS_FUSE};
    int removed = 0;
    double worst = equivalence_sweep(masks, 5, 300, &removed);
    printf("      300 random graphs x 5 pass sets: max |diff| = %g, ops removed = %d\n", worst,
           removed);
    CHECK(worst <= 1e-5);
    CHECK(removed > 100); /* the generator really does exercise the passes */
}


/* ---------------------------------------------------------------------------- */
/* Fusion                                                                       */
/* ---------------------------------------------------------------------------- */

static int count_op(const Graph *g, OpType op) {
    int c = 0;
    for (int i = 0; i < g->count; i++) c += !g->nodes[i].dead && g->nodes[i].op == op;
    return c;
}

static void fusion_on_mlp(void) {
    char err[512];
    Graph *g = parse_model_file("examples/tiny_mlp.txt", err, sizeof err);
    CHECK(g != NULL);
    CHECK(pass_fuse_linear(g) == 2);      /* both MATMUL+ADD pairs */
    CHECK(pass_fuse_linear_relu(g) == 1); /* the first layer's ReLU */
    CHECK(count_op(g, OP_LINEAR_RELU) == 1 && count_op(g, OP_LINEAR) == 1);
    CHECK(count_op(g, OP_MATMUL) == 0 && count_op(g, OP_ADD) == 0 && count_op(g, OP_RELU) == 0);
    CHECK(live_compute(g) == 3);
    graph_free(g);
}

static void fusion_skips_shared_matmul(void) {
    /* m feeds both ADD and SIGMOID, so MATMUL+ADD must not fuse. */
    char err[512];
    Graph *g = parse_model_file("examples/shared_value.txt", err, sizeof err);
    CHECK(g != NULL);
    CHECK(pass_fuse_linear(g) == 0);
    CHECK(count_op(g, OP_MATMUL) == 1);
    graph_free(g);
}

static void fusion_skips_shared_linear(void) {
    /* MATMUL+ADD may fuse (m has one consumer), but a = LINEAR is read by both
     * RELU and the final ADD, so LINEAR+RELU must not fuse. */
    Graph *g = parse("INPUT x 1 2\nCONST W 2 2 1 2 3 4\nCONST b 1 2 1 -9\n"
                     "MATMUL m x W\nADD a m b\nRELU r a\nADD y r a\nOUTPUT y\n");
    CHECK(g != NULL);
    CHECK(pass_fuse_linear(g) == 1);
    CHECK(pass_fuse_linear_relu(g) == 0);
    CHECK(count_op(g, OP_LINEAR) == 1 && count_op(g, OP_RELU) == 1);
    graph_free(g);
}

static void fusion_skips_matmul_that_is_output(void) {
    /* m is the graph output and also read by ADD. It must stay materialized. */
    Graph *g = parse("INPUT x 1 2\nCONST W 2 2 1 2 3 4\nCONST b 1 2 1 1\n"
                     "MATMUL m x W\nADD a m b\nOUTPUT m\n");
    CHECK(g != NULL);
    CHECK(pass_fuse_linear(g) == 0);
    graph_free(g);
}

static void fusion_bias_on_left(void) {
    /* ADD(b, MATMUL) with a full-shape b is also a LINEAR (IEEE + is commutative). */
    Graph *g = parse("INPUT x 2 2\nCONST W 2 2 1 2 3 4\nCONST b 2 2 1 1 2 2\n"
                     "MATMUL m x W\nADD a b m\nOUTPUT a\n");
    CHECK(g != NULL);
    Graph *ref = graph_clone(g);
    CHECK(pass_fuse_linear(g) == 1);
    int s[2] = {2, 2};
    float xv[4] = {1, -2, 3, 0.5f};
    Tensor *x = tensor_from(2, s, xv);
    char err[256];
    Tensor *y1 = graph_run(ref, x, 1, NULL, err, sizeof err);
    Tensor *y2 = graph_run(g, x, 1, NULL, err, sizeof err);
    CHECK(y1 && y2 && memcmp(y1->data, y2->data, 4 * sizeof(float)) == 0);
    tensor_free(x);
    tensor_free(y1);
    tensor_free(y2);
    graph_free(ref);
    graph_free(g);
}

static void fusion_skips_broadcast_matmul(void) {
    /* Here the MATMUL result (1 x 2) is the broadcast operand of a (3 x 2) ADD.
     * A fused LINEAR would only produce 1 row, so this must not fuse. */
    Graph *g = parse("INPUT x 1 2\nCONST W 2 2 1 2 3 4\nCONST big 3 2 1 2 3 4 5 6\n"
                     "MATMUL m x W\nADD a big m\nOUTPUT a\n");
    CHECK(g != NULL);
    CHECK(pass_fuse_linear(g) == 0);
    graph_free(g);
}

static void cleanup_enables_fusion(void) {
    /* In redundant.txt the second MATMUL has a duplicate feeding a dead SIGMOID.
     * Only after CSE + DCE does it have a single consumer, so "all" fuses more
     * than "fuse" alone would on the raw graph. */
    char err[512];
    Graph *g = parse_model_file("examples/redundant.txt", err, sizeof err);
    CHECK(g != NULL);
    PassReport r;
    CHECK(run_passes(g, PASS_ALL, &r, NULL, err, sizeof err) == 0);
    CHECK(live_compute(g) == 3); /* LINEAR_RELU, LINEAR, SOFTMAX */
    CHECK(count_op(g, OP_LINEAR_RELU) == 1 && count_op(g, OP_LINEAR) == 1);
    graph_free(g);
}

static void random_graphs_fusion_equivalent(void) {
    const unsigned masks[] = {PASS_FUSE, PASS_ALL};
    int removed = 0;
    double worst = equivalence_sweep(masks, 2, 300, &removed);
    printf("      300 random graphs x {fuse, all}: max |diff| = %g, ops removed = %d\n", worst,
           removed);
    CHECK(worst <= 1e-5);
    CHECK(removed > 100);
}

void test_passes(void) {
    RUN(dce_removes_unreachable);
    RUN(fold_folds_constant_chain);
    RUN(fold_whole_graph_to_constant);
    RUN(simplify_patterns);
    RUN(simplify_keeps_relu_after_sigmoid);
    RUN(simplify_ignores_non_identity);
    RUN(cse_merges_and_cascades);
    RUN(cse_dedupes_identical_constants);
    RUN(cse_respects_operand_order);
    RUN(pass_spec_parsing);
    RUN(redundant_example_end_to_end);
    RUN(random_graphs_cleanup_passes_equivalent);
    RUN(fusion_on_mlp);
    RUN(fusion_skips_shared_matmul);
    RUN(fusion_skips_shared_linear);
    RUN(fusion_skips_matmul_that_is_output);
    RUN(fusion_bias_on_left);
    RUN(fusion_skips_broadcast_matmul);
    RUN(cleanup_enables_fusion);
    RUN(random_graphs_fusion_equivalent);
}
