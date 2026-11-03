#ifndef TINYNNC_EXECUTOR_H
#define TINYNNC_EXECUTOR_H

/*
 * Executor.
 *
 * Execution has two phases, like a real runtime:
 *   plan_create()  "compile time": the topological order and the number of
 *                  consumers of each value. Computed once per graph.
 *   plan_run()     "run time": walks the order, allocates each node's output,
 *                  runs its kernel, and (if reuse is on) frees each
 *                  intermediate as soon as its last consumer has run.
 *
 * A plan is only valid for the exact graph it was created from. Re-create it
 * after running passes.
 */

#include "graph.h"

typedef struct ExecPlan ExecPlan;

typedef struct ExecStats {
    size_t peak_bytes;   /* peak bytes of activations (intermediates + output) during the run */
    int kernels_run;     /* number of kernel calls */
} ExecStats;

ExecPlan *plan_create(const Graph *g, char *err, size_t errlen);
void plan_free(ExecPlan *p);

/* Runs the graph on `input` (its shape must match the INPUT node). Returns a new
 * tensor owned by the caller, or NULL with a message in err.
 * reuse = 1: free each intermediate after its last use (liveness-based).
 * reuse = 0: keep every intermediate until the end (used to measure the savings). */
Tensor *plan_run(const ExecPlan *p, const Graph *g, const Tensor *input, int reuse,
                 ExecStats *stats, char *err, size_t errlen);

/* Convenience: plan_create + plan_run + plan_free. */
Tensor *graph_run(const Graph *g, const Tensor *input, int reuse, ExecStats *stats,
                  char *err, size_t errlen);

/* Runs one node's kernel: in[] are its input values, and out is pre-allocated with
 * the node's output shape. Constant folding uses this too, so compile-time
 * evaluation runs exactly the same code as run-time evaluation. Returns 0 or -1. */
int exec_node(const Node *n, const Tensor *const *in, Tensor *out);

#endif
