#include "passes.h"

#include <stdlib.h>
#include <string.h>

#include "executor.h"

/*
 * Every pass follows the same contract:
 *   - It only rewrites live nodes, and leaves the graph valid (acyclic, shapes correct).
 *   - The rewritten graph computes the same output as before. The safety condition
 *     that guarantees this is written above each pass.
 *   - It returns the number of nodes it changed (0 = nothing to do), or -1 on error.
 *
 * A pass that makes node u redundant redirects u's consumers to an equivalent
 * node (graph_replace_uses) and then removes u, which by then has no uses. The
 * inputs u used to read may now be unused as well, and DCE removes those.
 */

/* Live nodes in topological order. Caller frees. NULL on error. */
static int *topo(const Graph *g, int *n) {
    char err[256];
    int *order = malloc((size_t)(g->count ? g->count : 1) * sizeof(int));
    if (order && graph_topo_order(g, order, n, err, sizeof err) != 0) {
        free(order);
        return NULL;
    }
    return order;
}

static int is_live_op(const Graph *g, int id, OpType op) {
    return id >= 0 && !g->nodes[id].dead && g->nodes[id].op == op;
}

/* ------------------------------------------------------------------------- */
/* Dead code elimination                                                      */
/* ------------------------------------------------------------------------- */
/*
 * Removes every node that the output does not depend on.
 * Safe because: a node that cannot reach the output cannot change it. Our ops
 * have no side effects (no I/O, no state), so there is nothing else to preserve.
 * The INPUT node is always kept, because it is the graph's interface even if unused.
 * Method: depth-first search backwards from the output along input edges (a mark
 * phase), then kill everything that was not marked (a sweep phase).
 */
int pass_dce(Graph *g) {
    if (g->output < 0) return 0;
    char *mark = calloc((size_t)g->count, 1);
    int *stack = malloc((size_t)g->count * sizeof(int));
    if (!mark || !stack) {
        free(mark);
        free(stack);
        return -1;
    }
    int sp = 0, removed = 0;
    stack[sp++] = g->output;
    mark[g->output] = 1;
    while (sp > 0) {
        const Node *n = &g->nodes[stack[--sp]];
        for (int j = 0; j < n->ninputs; j++) {
            int p = n->inputs[j];
            if (!mark[p]) {
                mark[p] = 1;
                stack[sp++] = p; /* each node is pushed at most once, so sp < count */
            }
        }
    }
    for (int i = 0; i < g->count; i++) {
        Node *n = &g->nodes[i];
        if (!n->dead && !mark[i] && n->op != OP_INPUT) {
            graph_kill_node(g, i);
            removed++;
        }
    }
    free(mark);
    free(stack);
    return removed;
}

/* ------------------------------------------------------------------------- */
/* Constant folding                                                           */
/* ------------------------------------------------------------------------- */
/*
 * If every input of a node is a CONST, compute the node now and turn it into a
 * CONST holding the result.
 * Safe because: the result does not depend on the run-time input. It uses the
 * same kernel (exec_node) the executor would run, so the folded value is
 * bit-identical to what execution would have produced.
 * Processing in topological order folds whole constant chains in one sweep:
 * once a node becomes CONST, the nodes after it may become foldable.
 */
int pass_constant_fold(Graph *g) {
    int n = 0, folded = 0;
    int *order = topo(g, &n);
    if (order == NULL) return -1;
    for (int k = 0; k < n; k++) {
        Node *nd = &g->nodes[order[k]];
        if (nd->op == OP_INPUT || nd->op == OP_CONST) continue;
        int all_const = 1;
        const Tensor *ins[NODE_MAX_INPUTS] = {NULL, NULL, NULL};
        for (int j = 0; j < nd->ninputs; j++) {
            const Node *p = &g->nodes[nd->inputs[j]];
            if (p->op != OP_CONST) all_const = 0;
            ins[j] = p->value;
        }
        if (!all_const) continue;
        Tensor *v = tensor_new(nd->ndim, nd->shape);
        if (v == NULL || exec_node(nd, ins, v) != 0) {
            tensor_free(v);
            free(order);
            return -1;
        }
        nd->op = OP_CONST;
        nd->ninputs = 0;
        nd->value = v;
        folded++;
    }
    free(order);
    return folded;
}

/* ------------------------------------------------------------------------- */
/* Algebraic simplification                                                   */
/* ------------------------------------------------------------------------- */

static int const_is_zero(const Graph *g, int id) {
    if (!is_live_op(g, id, OP_CONST)) return 0;
    const Tensor *t = g->nodes[id].value;
    for (size_t i = 0; i < t->size; i++)
        if (t->data[i] != 0.0f) return 0; /* -0.0f == 0.0f, so -0 counts as zero */
    return 1;
}

static int const_is_identity(const Graph *g, int id) {
    if (!is_live_op(g, id, OP_CONST)) return 0;
    const Tensor *t = g->nodes[id].value;
    if (t->ndim != 2 || t->shape[0] != t->shape[1]) return 0;
    int n = t->shape[0];
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            if (t->data[(size_t)i * n + j] != (i == j ? 1.0f : 0.0f)) return 0;
    return 1;
}

static int same_shape(const Node *a, const Node *b) {
    return a->ndim == b->ndim && memcmp(a->shape, b->shape, sizeof a->shape) == 0;
}

/*
 * Peephole rewrites. Each one replaces node u with an existing node v that
 * already holds the same value:
 *
 *   RELU(RELU(v)) -> RELU(v)          relu is idempotent: max(max(v,0),0) == max(v,0)
 *   RELU(LINEAR_RELU(..)) -> LINEAR_RELU(..)   same reason
 *   ADD(v, 0) and ADD(0, v) -> v      when v already has the result's shape
 *   MATMUL(v, I) and MATMUL(I, v) -> v   I is a CONST identity matrix
 *   FLATTEN(v) -> v                   when v is already 2-D with the same shape
 *
 * Deliberately NOT done: RELU(SIGMOID(v)) -> SIGMOID(v). That holds for
 * every real number (sigmoid > 0), but not for NaN: RELU(NaN) = 0 while
 * SIGMOID(NaN) = NaN. A compiler must not change results unless it can prove
 * the rewrite is safe for every input, and this one is not.
 *
 * The identity-MATMUL rule assumes finite activations: with v = inf, v*0 = NaN,
 * so v*I would contain NaN while v does not. Every input this project
 * produces is finite (no inf/NaN can arise from finite inputs in an MLP of these
 * ops), and the README lists this assumption.
 */
int pass_simplify(Graph *g) {
    int n = 0, changed = 0;
    int *order = topo(g, &n);
    if (order == NULL) return -1;
    for (int k = 0; k < n; k++) {
        int u = order[k];
        const Node *nd = &g->nodes[u];
        int a = nd->ninputs > 0 ? nd->inputs[0] : -1;
        int b = nd->ninputs > 1 ? nd->inputs[1] : -1;
        int repl = -1;
        switch (nd->op) {
        case OP_RELU:
            if (is_live_op(g, a, OP_RELU) || is_live_op(g, a, OP_LINEAR_RELU)) repl = a;
            break;
        case OP_ADD:
            if (const_is_zero(g, b)) repl = a; /* a always has the result's shape */
            else if (const_is_zero(g, a) && same_shape(&g->nodes[b], nd)) repl = b;
            break;
        case OP_MATMUL:
            if (const_is_identity(g, b)) repl = a;
            else if (const_is_identity(g, a)) repl = b;
            break;
        case OP_FLATTEN:
            if (same_shape(&g->nodes[a], nd)) repl = a;
            break;
        default:
            break;
        }
        if (repl >= 0) {
            /* Consumers (and the graph output) now read repl directly, so u has no
             * uses left and is removed right away. If it stayed live, it would match
             * again on the next fixed-point round and be counted forever. Later nodes
             * in this sweep already see the redirected edge, so chains like
             * RELU(RELU(RELU(v))) collapse in a single sweep. */
            graph_replace_uses(g, u, repl);
            graph_kill_node(g, u);
            changed++;
        }
    }
    free(order);
    return changed;
}

/* ------------------------------------------------------------------------- */
/* Common subexpression elimination                                           */
/* ------------------------------------------------------------------------- */

static int nodes_equivalent(const Graph *g, int x, int y) {
    const Node *a = &g->nodes[x], *b = &g->nodes[y];
    if (a->op != b->op || a->ninputs != b->ninputs || !same_shape(a, b)) return 0;
    if (a->op == OP_INPUT) return 0;
    if (a->op == OP_CONST) /* same bits -> same constant (dedupes repeated weights) */
        return memcmp(a->value->data, b->value->data, a->value->size * sizeof(float)) == 0;
    for (int j = 0; j < a->ninputs; j++)
        if (a->inputs[j] != b->inputs[j]) return 0;
    return 1;
}

/*
 * Two nodes with the same op and the same inputs compute the same value, so one
 * can be replaced by the other.
 * Safe because: every op is a pure function of its inputs (no randomness, no state).
 * Visiting in topological order makes it cascade. After two MATMULs merge, two ADDs
 * that read them now have identical inputs too, and merge when we reach them.
 * Operand order is compared exactly, so ADD(a,b) and ADD(b,a) are not merged.
 * This is O(n^2) in the number of nodes. That is fine at our graph sizes, and a
 * hash of (op, inputs) would make it O(n) for big graphs.
 */
int pass_cse(Graph *g) {
    int n = 0, merged = 0;
    int *order = topo(g, &n);
    if (order == NULL) return -1;
    for (int k = 0; k < n; k++) {
        int u = order[k];
        if (g->nodes[u].dead || g->nodes[u].op == OP_INPUT) continue;
        for (int i = 0; i < k; i++) {
            int v = order[i];
            if (g->nodes[v].dead || !nodes_equivalent(g, u, v)) continue;
            graph_replace_uses(g, u, v);
            graph_kill_node(g, u); /* u now has no consumers */
            merged++;
            break;
        }
    }
    free(order);
    return merged;
}

/* ------------------------------------------------------------------------- */
/* Pass manager                                                               */
/* ------------------------------------------------------------------------- */

int passes_parse(const char *spec, unsigned *mask, char *err, size_t errlen) {
    static const struct { const char *name; unsigned bit; } k_names[] = {
        {"fold", PASS_FOLD}, {"simplify", PASS_SIMPLIFY}, {"cse", PASS_CSE},
        {"dce", PASS_DCE},   {"fuse", PASS_FUSE},
    };
    *mask = 0;
    if (strcmp(spec, "none") == 0) return 0;
    if (strcmp(spec, "all") == 0) {
        *mask = PASS_ALL;
        return 0;
    }
    const char *p = spec;
    while (*p) {
        size_t len = strcspn(p, ",");
        int found = 0;
        for (size_t i = 0; i < sizeof k_names / sizeof k_names[0]; i++) {
            if (strlen(k_names[i].name) == len && strncmp(p, k_names[i].name, len) == 0) {
                *mask |= k_names[i].bit;
                found = 1;
            }
        }
        if (!found) {
            snprintf(err, errlen,
                     "unknown pass '%.*s' in --opt (use none, all, or fold,simplify,cse,dce,fuse)",
                     (int)len, p);
            return -1;
        }
        p += len;
        if (*p == ',') p++;
    }
    return 0;
}

static int run_one(int (*pass)(Graph *), Graph *g, int *total, int *changed) {
    int c = pass(g);
    if (c < 0) return -1;
    *total += c;
    *changed += c;
    return 0;
}

static void print_summary(FILE *log, const GraphStats *b, const GraphStats *a) {
    fprintf(log, "  %-30s %12s %12s\n", "", "before", "after");
    fprintf(log, "  %-30s %12d %12d\n", "compute ops", b->compute_nodes, a->compute_nodes);
    fprintf(log, "  %-30s %12d %12d\n", "live nodes (incl. consts)", b->live_nodes,
            a->live_nodes);
    fprintf(log, "  %-30s %12.0f %12.0f\n", "estimated FLOPs", b->flops, a->flops);
    fprintf(log, "  %-30s %12zu %12zu\n", "activation bytes (no reuse)", b->act_bytes,
            a->act_bytes);
    fprintf(log, "  %-30s %12zu %12zu\n", "constant bytes", b->const_bytes, a->const_bytes);
}

int run_passes(Graph *g, unsigned mask, PassReport *report, FILE *log, char *err,
               size_t errlen) {
    PassReport r;
    memset(&r, 0, sizeof r);
    GraphStats before, after;
    graph_stats(g, &before);

    /* The cleanup passes feed each other. Folding can create zero constants
     * for simplify, simplify can create duplicates for CSE, and all of them leave
     * dead nodes for DCE. So repeat them until a round changes nothing (a
     * fixed point). The cap is a guard. Every rewrite removes or freezes a
     * node, so the loop always ends well before it. */
    const int max_rounds = 16;
    int changed;
    do {
        changed = 0;
        if ((mask & PASS_FOLD) && run_one(pass_constant_fold, g, &r.fold, &changed)) goto fail;
        if ((mask & PASS_SIMPLIFY) && run_one(pass_simplify, g, &r.simplify, &changed))
            goto fail;
        if ((mask & PASS_CSE) && run_one(pass_cse, g, &r.cse, &changed)) goto fail;
        if ((mask & PASS_DCE) && run_one(pass_dce, g, &r.dce, &changed)) goto fail;
        r.rounds++;
    } while (changed && r.rounds < max_rounds);

    /* Sanity check: every pass must leave a well-formed graph. */
    if (graph_infer_shapes(g, err, errlen) != 0) return -1;

    graph_stats(g, &after);
    if (log && mask) {
        fprintf(log, "== optimization passes (%d round%s) ==\n", r.rounds,
                r.rounds == 1 ? "" : "s");
        if (mask & PASS_FOLD) fprintf(log, "  constant folding      : %d nodes folded\n", r.fold);
        if (mask & PASS_SIMPLIFY)
            fprintf(log, "  algebraic simplify    : %d nodes rewritten\n", r.simplify);
        if (mask & PASS_CSE) fprintf(log, "  common subexpr elim   : %d nodes merged\n", r.cse);
        if (mask & PASS_DCE) fprintf(log, "  dead code elimination : %d nodes removed\n", r.dce);
        print_summary(log, &before, &after);
    }
    if (report) *report = r;
    return 0;
fail:
    snprintf(err, errlen, "an optimization pass failed (out of memory or invalid graph)");
    return -1;
}
