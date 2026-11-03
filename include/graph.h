#ifndef TINYNNC_GRAPH_H
#define TINYNNC_GRAPH_H

/*
 * Graph IR.
 *
 * A model is a DAG of Nodes. Each node produces exactly one tensor value.
 * Edges are stored on the consumer side: node->inputs[] holds the ids of the
 * producer nodes. A node id is its index in graph->nodes, and ids never change.
 * Passes do not delete nodes. They set node->dead = 1, so every id held by a
 * pass, the executor, or a test stays valid.
 *
 * A node may have many consumers. "How many consumers does this value have?"
 * is computed on demand with graph_count_uses(). Passes need that answer
 * (fusion must not fuse a value that someone else still reads), and computing
 * it fresh means it can never be out of date after a rewrite.
 *
 * Ownership: the Graph owns its node array and each CONST node's Tensor.
 */

#include <stddef.h>
#include <stdio.h>

#include "tensor.h"

typedef enum {
    OP_INPUT,       /* graph input, value supplied at run time */
    OP_CONST,       /* compile-time constant (weights, biases, folded values) */
    OP_MATMUL,      /* (M x K) * (K x N) */
    OP_ADD,         /* (M x N) + (M x N | 1 x N) */
    OP_RELU,
    OP_SIGMOID,
    OP_SOFTMAX,     /* row-wise over the last dimension */
    OP_FLATTEN,     /* (d0, d1, ...) -> (d0, d1*d2*...) */
    OP_LINEAR,      /* fused MATMUL + ADD: inputs (x, W, b) */
    OP_LINEAR_RELU, /* fused MATMUL + ADD + RELU: inputs (x, W, b) */
    OP_COUNT
} OpType;

#define NODE_MAX_INPUTS 3
#define NODE_NAME_LEN 32
#define GRAPH_MAX_NODES 100000

typedef struct Node {
    int id;
    OpType op;
    char name[NODE_NAME_LEN];
    int ninputs;
    int inputs[NODE_MAX_INPUTS];   /* producer node ids */
    int ndim;                      /* output shape. 0 until shape inference has run */
    int shape[TENSOR_MAX_DIMS];
    Tensor *value;                 /* CONST only: owned constant data */
    int dead;                      /* removed by a pass. Ignored everywhere */
    int line;                      /* source line in the model file (0 = created by a pass) */
} Node;

typedef struct Graph {
    Node *nodes;
    int count;
    int cap;
    int output;                    /* id of the output node, -1 if not set */
} Graph;

const char *op_name(OpType op);
int op_from_name(const char *name, OpType *out);  /* 1 if found */
int op_arity(OpType op);                          /* number of inputs */

Graph *graph_new(void);
void graph_free(Graph *g);
Graph *graph_clone(const Graph *g);              /* deep copy, including constants */

/* Builders return the new node id, or -1 on error (bad args, OOM, too many nodes). */
int graph_add_input(Graph *g, const char *name, int ndim, const int *shape);
int graph_add_const(Graph *g, const char *name, Tensor *value); /* takes ownership, even on failure */
int graph_add_op(Graph *g, OpType op, const char *name, const int *inputs, int ninputs);

int graph_find(const Graph *g, const char *name); /* live node by name, or -1 */
int graph_input_id(const Graph *g);               /* first live INPUT, or -1 */

/* uses[i] = number of edges reading node i from live nodes, plus 1 if node i is
 * the graph output. `uses` must have room for g->count ints. */
void graph_count_uses(const Graph *g, int *uses);

/* Kahn's algorithm. Fills order[] (room for g->count ints) with the live node
 * ids, producers before consumers, and sets *n. Returns 0 on success, or -1 with
 * a message in err if there is a cycle or an edge points to a dead or invalid node. */
int graph_topo_order(const Graph *g, int *order, int *n, char *err, size_t errlen);

/* Computes node `id`'s output shape from its inputs' shapes (which must already
 * be known). Returns 0, or -1 with a readable message. */
int graph_infer_node(Graph *g, int id, char *err, size_t errlen);

/* Runs shape inference over the whole graph in topological order. */
int graph_infer_shapes(Graph *g, char *err, size_t errlen);

/* Redirects every edge (and the graph output) from old_id to new_id. */
void graph_replace_uses(Graph *g, int old_id, int new_id);

/* Marks a node dead and releases its constant data. */
void graph_kill_node(Graph *g, int id);

typedef struct GraphStats {
    int live_nodes;       /* all live nodes, including INPUT and CONST */
    int compute_nodes;    /* live nodes that run a kernel (not INPUT/CONST) */
    double flops;         /* estimated floating point ops for one run */
    size_t const_bytes;   /* bytes of constant data (weights) */
    size_t act_bytes;     /* sum of all compute outputs = memory if nothing is freed early */
} GraphStats;

double node_flops(const Graph *g, int id);
void graph_stats(const Graph *g, GraphStats *s);

void graph_print(const Graph *g, FILE *out);
void graph_to_dot(const Graph *g, FILE *out, const char *title);

#endif
