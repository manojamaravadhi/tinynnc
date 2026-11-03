#include "graph.h"

#include <stdlib.h>
#include <string.h>

static const char *const k_op_names[OP_COUNT] = {
    "INPUT", "CONST", "MATMUL", "ADD", "RELU", "SIGMOID",
    "SOFTMAX", "FLATTEN", "LINEAR", "LINEAR_RELU",
};
static const int k_op_arity[OP_COUNT] = {0, 0, 2, 2, 1, 1, 1, 1, 3, 3};

const char *op_name(OpType op) {
    return ((int)op >= 0 && op < OP_COUNT) ? k_op_names[op] : "?";
}

int op_arity(OpType op) { return ((int)op >= 0 && op < OP_COUNT) ? k_op_arity[op] : -1; }

int op_from_name(const char *name, OpType *out) {
    for (int i = 0; i < OP_COUNT; i++) {
        if (strcmp(name, k_op_names[i]) == 0) {
            *out = (OpType)i;
            return 1;
        }
    }
    return 0;
}

Graph *graph_new(void) {
    Graph *g = calloc(1, sizeof *g);
    if (g) g->output = -1;
    return g;
}

void graph_free(Graph *g) {
    if (g == NULL) return;
    for (int i = 0; i < g->count; i++) tensor_free(g->nodes[i].value);
    free(g->nodes);
    free(g);
}

Graph *graph_clone(const Graph *g) {
    if (g == NULL) return NULL;
    Graph *c = graph_new();
    if (c == NULL) return NULL;
    if (g->count > 0) {
        c->nodes = malloc((size_t)g->count * sizeof(Node));
        if (c->nodes == NULL) {
            free(c);
            return NULL;
        }
        memcpy(c->nodes, g->nodes, (size_t)g->count * sizeof(Node));
    }
    c->count = c->cap = g->count;
    c->output = g->output;
    /* The memcpy copied the CONST pointers too. Replace each with a deep copy,
     * so the two graphs never share (or double-free) a tensor. */
    for (int i = 0; i < c->count; i++) {
        if (g->nodes[i].value == NULL) continue;
        c->nodes[i].value = tensor_clone(g->nodes[i].value);
        if (c->nodes[i].value == NULL) {
            for (int j = i + 1; j < c->count; j++) c->nodes[j].value = NULL;
            graph_free(c);
            return NULL;
        }
    }
    return c;
}

/* Appends a blank node and returns its id. The node array may be reallocated,
 * so callers must not hold Node pointers across this call. */
static int graph_push(Graph *g, OpType op, const char *name) {
    if (g == NULL || g->count >= GRAPH_MAX_NODES) return -1;
    if (g->count == g->cap) {
        int ncap = g->cap ? g->cap * 2 : 16;
        Node *nn = realloc(g->nodes, (size_t)ncap * sizeof(Node));
        if (nn == NULL) return -1;
        g->nodes = nn;
        g->cap = ncap;
    }
    Node *n = &g->nodes[g->count];
    memset(n, 0, sizeof *n);
    n->id = g->count;
    n->op = op;
    snprintf(n->name, sizeof n->name, "%s", name ? name : "");
    return g->count++;
}

int graph_add_input(Graph *g, const char *name, int ndim, const int *shape) {
    if (shape_numel(ndim, shape) == 0) return -1;
    int id = graph_push(g, OP_INPUT, name);
    if (id < 0) return -1;
    Node *n = &g->nodes[id];
    n->ndim = ndim;
    for (int i = 0; i < TENSOR_MAX_DIMS; i++) n->shape[i] = i < ndim ? shape[i] : 1;
    return id;
}

int graph_add_const(Graph *g, const char *name, Tensor *value) {
    if (value == NULL) return -1;
    int id = graph_push(g, OP_CONST, name);
    if (id < 0) {
        tensor_free(value);
        return -1;
    }
    Node *n = &g->nodes[id];
    n->value = value;
    n->ndim = value->ndim;
    memcpy(n->shape, value->shape, sizeof n->shape);
    return id;
}

int graph_add_op(Graph *g, OpType op, const char *name, const int *inputs, int ninputs) {
    if (g == NULL || op == OP_INPUT || op == OP_CONST || op_arity(op) != ninputs) return -1;
    for (int i = 0; i < ninputs; i++)
        if (inputs[i] < 0 || inputs[i] >= g->count || g->nodes[inputs[i]].dead) return -1;
    int id = graph_push(g, op, name);
    if (id < 0) return -1;
    Node *n = &g->nodes[id];
    n->ninputs = ninputs;
    for (int i = 0; i < ninputs; i++) n->inputs[i] = inputs[i];
    return id;
}

int graph_find(const Graph *g, const char *name) {
    for (int i = g->count - 1; i >= 0; i--)
        if (!g->nodes[i].dead && strcmp(g->nodes[i].name, name) == 0) return i;
    return -1;
}

int graph_input_id(const Graph *g) {
    for (int i = 0; i < g->count; i++)
        if (!g->nodes[i].dead && g->nodes[i].op == OP_INPUT) return i;
    return -1;
}

void graph_count_uses(const Graph *g, int *uses) {
    memset(uses, 0, (size_t)g->count * sizeof(int));
    for (int i = 0; i < g->count; i++) {
        const Node *n = &g->nodes[i];
        if (n->dead) continue;
        for (int j = 0; j < n->ninputs; j++) uses[n->inputs[j]]++;
    }
    /* The graph output counts as one consumer, so a pass can never decide that
     * the final result is "unused", or fuse it away. */
    if (g->output >= 0) uses[g->output]++;
}

int graph_topo_order(const Graph *g, int *order, int *n_out, char *err, size_t errlen) {
    int n = g->count, live = 0, done = 0, rc = -1;
    *n_out = 0;
    /* Before sorting, check that every edge points at a live node in range.
     * Otherwise a bad rewrite would show up later as a crash instead of an error. */
    for (int i = 0; i < n; i++) {
        const Node *nd = &g->nodes[i];
        if (nd->dead) continue;
        live++;
        for (int j = 0; j < nd->ninputs; j++) {
            int p = nd->inputs[j];
            if (p < 0 || p >= n || g->nodes[p].dead) {
                snprintf(err, errlen, "node '%s' reads an invalid or removed node (id %d)",
                         nd->name, p);
                return -1;
            }
        }
    }
    if (n == 0) return 0;

    /* Kahn's algorithm. indeg[v] counts the not-yet-scheduled inputs of v. The
     * consumer lists are stored in CSR form (offsets + flat array), so the sort
     * runs in O(nodes + edges) instead of rescanning the whole graph for every node. */
    int *indeg = calloc((size_t)n, sizeof(int));
    int *off = calloc((size_t)n + 1, sizeof(int));
    int *adj = malloc(((size_t)n * NODE_MAX_INPUTS + 1) * sizeof(int));
    int *fill = calloc((size_t)n, sizeof(int));
    if (!indeg || !off || !adj || !fill) {
        snprintf(err, errlen, "out of memory in topological sort");
        goto out;
    }
    for (int i = 0; i < n; i++) {
        const Node *nd = &g->nodes[i];
        if (nd->dead) continue;
        indeg[i] = nd->ninputs;
        for (int j = 0; j < nd->ninputs; j++) off[nd->inputs[j] + 1]++;
    }
    for (int i = 0; i < n; i++) off[i + 1] += off[i];
    for (int i = 0; i < n; i++) {
        const Node *nd = &g->nodes[i];
        if (nd->dead) continue;
        for (int j = 0; j < nd->ninputs; j++) {
            int p = nd->inputs[j];
            adj[off[p] + fill[p]++] = i;
        }
    }
    /* order[] doubles as the FIFO queue: entries before `done` are finished,
     * entries between `done` and `tail` are ready to run. */
    int tail = 0;
    for (int i = 0; i < n; i++)
        if (!g->nodes[i].dead && indeg[i] == 0) order[tail++] = i;
    while (done < tail) {
        int u = order[done++];
        for (int k = off[u]; k < off[u + 1]; k++)
            if (--indeg[adj[k]] == 0) order[tail++] = adj[k];
    }
    if (done < live) {
        /* Nodes that never reached indegree 0 are on a cycle, or depend on one. */
        for (int i = 0; i < n; i++) {
            if (!g->nodes[i].dead && indeg[i] > 0) {
                snprintf(err, errlen, "cycle detected: node '%s' (%s) depends on itself",
                         g->nodes[i].name, op_name(g->nodes[i].op));
                break;
            }
        }
        goto out;
    }
    *n_out = done;
    rc = 0;
out:
    free(indeg);
    free(off);
    free(adj);
    free(fill);
    return rc;
}

static const char *shp(const Node *n, char *buf, size_t len) {
    return shape_str(n->ndim, n->shape, buf, len);
}

static void set_shape2(Node *n, int r, int c) {
    n->ndim = 2;
    n->shape[0] = r;
    n->shape[1] = c;
    n->shape[2] = n->shape[3] = 1;
}

/* Shared check for MATMUL and LINEAR: x (M x K) times W (K x N). */
static int check_matmul(const Node *n, const Node *x, const Node *w, char *err, size_t errlen) {
    char a[48], b[48];
    if (x->ndim != 2 || w->ndim != 2) {
        snprintf(err, errlen, "%s '%s': operands must be 2-D, got '%s' %s and '%s' %s",
                 op_name(n->op), n->name, x->name, shp(x, a, 48), w->name, shp(w, b, 48));
        return -1;
    }
    if (x->shape[1] != w->shape[0]) {
        snprintf(err, errlen,
                 "%s '%s': inner dimensions differ: '%s' is %s, '%s' is %s (%d != %d)",
                 op_name(n->op), n->name, x->name, shp(x, a, 48), w->name, shp(w, b, 48),
                 x->shape[1], w->shape[0]);
        return -1;
    }
    return 0;
}

/* Checks that `b` can be added to an (M x N) value: either (M x N), or (1 x N) broadcast. */
static int check_bias(const Node *n, int M, int N, const Node *b, char *err, size_t errlen) {
    char a[48];
    if (b->ndim != 2 || b->shape[1] != N || (b->shape[0] != 1 && b->shape[0] != M)) {
        snprintf(err, errlen,
                 "%s '%s': operand '%s' has shape %s, expected [%dx%d] or [1x%d]",
                 op_name(n->op), n->name, b->name, shp(b, a, 48), M, N, N);
        return -1;
    }
    return 0;
}

int graph_infer_node(Graph *g, int id, char *err, size_t errlen) {
    Node *n = &g->nodes[id];
    const Node *in[NODE_MAX_INPUTS] = {NULL, NULL, NULL};
    for (int j = 0; j < n->ninputs; j++) {
        in[j] = &g->nodes[n->inputs[j]];
        if (in[j]->ndim == 0) {
            snprintf(err, errlen, "node '%s': input '%s' has no shape yet", n->name, in[j]->name);
            return -1;
        }
    }
    char a[48];
    switch (n->op) {
    case OP_INPUT:
    case OP_CONST:
        return n->ndim > 0 ? 0 : -1;
    case OP_MATMUL:
        if (check_matmul(n, in[0], in[1], err, errlen)) return -1;
        set_shape2(n, in[0]->shape[0], in[1]->shape[1]);
        return 0;
    case OP_ADD:
        /* The first operand sets the result shape. The second may broadcast a
         * single row. This matches what op_add supports. */
        if (in[0]->ndim != 2) {
            snprintf(err, errlen, "ADD '%s': operand '%s' must be 2-D, got %s", n->name,
                     in[0]->name, shp(in[0], a, 48));
            return -1;
        }
        if (check_bias(n, in[0]->shape[0], in[0]->shape[1], in[1], err, errlen)) return -1;
        set_shape2(n, in[0]->shape[0], in[0]->shape[1]);
        return 0;
    case OP_LINEAR:
    case OP_LINEAR_RELU:
        if (check_matmul(n, in[0], in[1], err, errlen)) return -1;
        if (check_bias(n, in[0]->shape[0], in[1]->shape[1], in[2], err, errlen)) return -1;
        set_shape2(n, in[0]->shape[0], in[1]->shape[1]);
        return 0;
    case OP_RELU:
    case OP_SIGMOID:
    case OP_SOFTMAX:
        n->ndim = in[0]->ndim;
        memcpy(n->shape, in[0]->shape, sizeof n->shape);
        return 0;
    case OP_FLATTEN: {
        size_t total = shape_numel(in[0]->ndim, in[0]->shape);
        if (in[0]->ndim == 1) set_shape2(n, 1, (int)total);
        else set_shape2(n, in[0]->shape[0], (int)(total / (size_t)in[0]->shape[0]));
        return 0;
    }
    default:
        snprintf(err, errlen, "node '%s': unknown op %d", n->name, (int)n->op);
        return -1;
    }
}

int graph_infer_shapes(Graph *g, char *err, size_t errlen) {
    int *order = malloc((size_t)(g->count ? g->count : 1) * sizeof(int));
    int n = 0, rc = -1;
    if (order == NULL) {
        snprintf(err, errlen, "out of memory");
        return -1;
    }
    if (graph_topo_order(g, order, &n, err, errlen) == 0) {
        rc = 0;
        for (int k = 0; k < n && rc == 0; k++) rc = graph_infer_node(g, order[k], err, errlen);
    }
    free(order);
    return rc;
}

void graph_replace_uses(Graph *g, int old_id, int new_id) {
    for (int i = 0; i < g->count; i++) {
        Node *n = &g->nodes[i];
        if (n->dead) continue;
        for (int j = 0; j < n->ninputs; j++)
            if (n->inputs[j] == old_id) n->inputs[j] = new_id;
    }
    if (g->output == old_id) g->output = new_id;
}

void graph_kill_node(Graph *g, int id) {
    Node *n = &g->nodes[id];
    n->dead = 1;
    tensor_free(n->value);
    n->value = NULL;
}

/* FLOP estimates: a multiply-add counts as 2, and each elementwise op counts as 1 per
 * element. Sigmoid (exp, add, divide) counts as 3 and softmax (sub, exp, sum, scale)
 * as 4 per element. These are estimates, used to compare graphs, not to predict time. */
double node_flops(const Graph *g, int id) {
    const Node *n = &g->nodes[id];
    double out = (double)shape_numel(n->ndim, n->shape);
    double M = n->shape[0], N = n->shape[1];
    switch (n->op) {
    case OP_MATMUL: return 2.0 * M * g->nodes[n->inputs[0]].shape[1] * N;
    case OP_LINEAR: return 2.0 * M * g->nodes[n->inputs[0]].shape[1] * N + out;
    case OP_LINEAR_RELU: return 2.0 * M * g->nodes[n->inputs[0]].shape[1] * N + 2.0 * out;
    case OP_ADD:
    case OP_RELU: return out;
    case OP_SIGMOID: return 3.0 * out;
    case OP_SOFTMAX: return 4.0 * out;
    default: return 0.0; /* INPUT, CONST, FLATTEN (a copy) */
    }
}

void graph_stats(const Graph *g, GraphStats *s) {
    memset(s, 0, sizeof *s);
    for (int i = 0; i < g->count; i++) {
        const Node *n = &g->nodes[i];
        if (n->dead) continue;
        s->live_nodes++;
        if (n->op == OP_CONST) {
            s->const_bytes += tensor_bytes(n->value);
        } else if (n->op != OP_INPUT) {
            s->compute_nodes++;
            s->flops += node_flops(g, i);
            s->act_bytes += shape_numel(n->ndim, n->shape) * sizeof(float);
        }
    }
}

void graph_print(const Graph *g, FILE *out) {
    int *order = malloc((size_t)(g->count ? g->count : 1) * sizeof(int));
    int n = 0;
    char err[256], sb[48];
    if (order == NULL) return;
    /* Print in execution order. If the graph is broken (for example, has a cycle),
     * fall back to id order so the printer still shows something useful. */
    if (graph_topo_order(g, order, &n, err, sizeof err) != 0) {
        fprintf(out, "  (invalid graph: %s; printing in id order)\n", err);
        n = 0;
        for (int i = 0; i < g->count; i++)
            if (!g->nodes[i].dead) order[n++] = i;
    }
    for (int k = 0; k < n; k++) {
        const Node *nd = &g->nodes[order[k]];
        char args[128] = "";
        size_t used = 0;
        for (int j = 0; j < nd->ninputs && used < sizeof args; j++) {
            int w = snprintf(args + used, sizeof args - used, "%s%s", j ? ", " : "",
                             g->nodes[nd->inputs[j]].name);
            if (w > 0) used += (size_t)w;
        }
        char call[160];
        if (nd->ninputs) snprintf(call, sizeof call, "%s(%s)", op_name(nd->op), args);
        else snprintf(call, sizeof call, "%s", op_name(nd->op));
        fprintf(out, "  %%%-3d %-10s = %-34s %s%s\n", nd->id, nd->name, call,
                nd->ndim ? shape_str(nd->ndim, nd->shape, sb, sizeof sb) : "[?]",
                nd->id == g->output ? "   <- output" : "");
    }
    free(order);
}

void graph_to_dot(const Graph *g, FILE *out, const char *title) {
    char sb[48];
    fprintf(out, "digraph \"%s\" {\n", title ? title : "tinynnc");
    fprintf(out, "  rankdir=TB;\n  node [shape=box, style=\"rounded,filled\", "
                 "fontname=\"Helvetica\", fillcolor=\"#ffffff\"];\n");
    for (int i = 0; i < g->count; i++) {
        const Node *n = &g->nodes[i];
        if (n->dead) continue;
        const char *color = "#ffffff";
        if (n->op == OP_INPUT) color = "#cfe8ff";
        else if (n->op == OP_CONST) color = "#e5e7eb";
        else if (n->op == OP_LINEAR || n->op == OP_LINEAR_RELU) color = "#fde68a"; /* fused */
        fprintf(out, "  n%d [label=\"%s\\n%s %s\", fillcolor=\"%s\"%s];\n", i, n->name,
                op_name(n->op), shape_str(n->ndim, n->shape, sb, sizeof sb), color,
                n->op == OP_CONST ? ", shape=note" : "");
    }
    for (int i = 0; i < g->count; i++) {
        const Node *n = &g->nodes[i];
        if (n->dead) continue;
        for (int j = 0; j < n->ninputs; j++) fprintf(out, "  n%d -> n%d;\n", n->inputs[j], i);
    }
    if (g->output >= 0)
        fprintf(out, "  out [shape=plaintext, style=\"\", label=\"output\"];\n  n%d -> out;\n",
                g->output);
    fprintf(out, "}\n");
}
