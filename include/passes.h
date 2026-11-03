#ifndef TINYNNC_PASSES_H
#define TINYNNC_PASSES_H

/*
 * Optimization passes. Each pass rewrites the graph in place and returns how many
 * nodes it changed (or -1 on an internal error such as out of memory).
 * Safety conditions for each pass are documented in docs/DESIGN.md and above each
 * function in passes.c.
 */

#include <stdio.h>

#include "graph.h"

enum {
    PASS_FOLD = 1u << 0,      /* constant folding */
    PASS_SIMPLIFY = 1u << 1,  /* algebraic simplification / redundant op removal */
    PASS_CSE = 1u << 2,       /* common subexpression elimination */
    PASS_DCE = 1u << 3,       /* dead code elimination */
    PASS_FUSE = 1u << 4,      /* MATMUL+ADD -> LINEAR, LINEAR+RELU -> LINEAR_RELU */
    PASS_ALL = 0x1f
};

/* "none", "all", or a comma list of: fold, simplify, cse, dce, fuse. */
int passes_parse(const char *spec, unsigned *mask, char *err, size_t errlen);

int pass_constant_fold(Graph *g);
int pass_simplify(Graph *g);
int pass_cse(Graph *g);
int pass_dce(Graph *g);
int pass_fuse_linear(Graph *g);
int pass_fuse_linear_relu(Graph *g);

typedef struct PassReport {
    int fold, simplify, cse, dce, fuse; /* total nodes changed by each pass */
    int rounds;                         /* fixed-point iterations of the cleanup passes */
} PassReport;

/* Runs the enabled passes. fold/simplify/cse/dce repeat until nothing changes
 * (a fixed point), because one pass can expose work for another. Fusion runs
 * last, on the cleaned-up graph. If log is not NULL, prints a per-pass report and
 * a before/after summary. Returns 0, or -1 with a message. */
int run_passes(Graph *g, unsigned mask, PassReport *report, FILE *log, char *err,
               size_t errlen);

#endif
