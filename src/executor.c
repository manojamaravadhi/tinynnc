#include "executor.h"

#include <stdlib.h>
#include <string.h>

#include "ops.h"

struct ExecPlan {
    int *order;   /* live node ids, producers before consumers */
    int norder;
    int *uses;    /* consumers per node (+1 for the graph output) */
    int count;    /* g->count when the plan was made */
    int input_id;
    int output_id;
};

int exec_node(const Node *n, const Tensor *const *in, Tensor *out) {
    switch (n->op) {
    case OP_MATMUL:
        op_matmul(in[0]->data, in[1]->data, out->data, in[0]->shape[0], in[0]->shape[1],
                  in[1]->shape[1]);
        return 0;
    case OP_ADD:
        op_add(in[0]->data, in[1]->data, out->data, out->shape[0], out->shape[1],
               in[1]->shape[0]);
        return 0;
    case OP_RELU:
        op_relu(in[0]->data, out->data, out->size);
        return 0;
    case OP_SIGMOID:
        op_sigmoid(in[0]->data, out->data, out->size);
        return 0;
    case OP_SOFTMAX: {
        int cols = out->shape[out->ndim - 1];
        op_softmax(in[0]->data, out->data, (int)(out->size / (size_t)cols), cols);
        return 0;
    }
    case OP_FLATTEN:
        op_copy(in[0]->data, out->data, out->size);
        return 0;
    case OP_LINEAR:
        op_linear(in[0]->data, in[1]->data, in[2]->data, out->data, in[0]->shape[0],
                  in[0]->shape[1], in[1]->shape[1], in[2]->shape[0]);
        return 0;
    case OP_LINEAR_RELU:
        op_linear_relu(in[0]->data, in[1]->data, in[2]->data, out->data, in[0]->shape[0],
                       in[0]->shape[1], in[1]->shape[1], in[2]->shape[0]);
        return 0;
    default:
        return -1; /* INPUT and CONST have no kernel */
    }
}

ExecPlan *plan_create(const Graph *g, char *err, size_t errlen) {
    if (g->output < 0 || g->output >= g->count || g->nodes[g->output].dead) {
        snprintf(err, errlen, "graph has no output");
        return NULL;
    }
    ExecPlan *p = calloc(1, sizeof *p);
    if (p == NULL) goto oom;
    p->count = g->count;
    p->output_id = g->output;
    p->input_id = graph_input_id(g);
    p->order = malloc((size_t)g->count * sizeof(int));
    p->uses = malloc((size_t)g->count * sizeof(int));
    if (!p->order || !p->uses) goto oom;
    if (graph_topo_order(g, p->order, &p->norder, err, errlen) != 0) {
        plan_free(p);
        return NULL;
    }
    graph_count_uses(g, p->uses);
    return p;
oom:
    snprintf(err, errlen, "out of memory creating plan");
    plan_free(p);
    return NULL;
}

void plan_free(ExecPlan *p) {
    if (p == NULL) return;
    free(p->order);
    free(p->uses);
    free(p);
}

Tensor *plan_run(const ExecPlan *p, const Graph *g, const Tensor *input, int reuse,
                 ExecStats *stats, char *err, size_t errlen) {
    if (p->input_id >= 0) {
        const Node *in = &g->nodes[p->input_id];
        if (input == NULL || input->ndim != in->ndim ||
            memcmp(input->shape, in->shape, sizeof in->shape) != 0) {
            char want[48], got[48] = "none";
            if (input) shape_str(input->ndim, input->shape, got, sizeof got);
            snprintf(err, errlen, "input shape %s does not match INPUT '%s' %s", got, in->name,
                     shape_str(in->ndim, in->shape, want, sizeof want));
            return NULL;
        }
    }
    int n = p->count;
    /* vals[i] = the value of node i (borrowed or owned). own[i] is set only for
     * buffers this run allocated, so we never free the caller's input or a
     * graph constant. rem[i] = consumers of i that have not run yet. */
    const Tensor **vals = calloc((size_t)n, sizeof *vals);
    Tensor **own = calloc((size_t)n, sizeof *own);
    int *rem = malloc((size_t)n * sizeof(int));
    Tensor *result = NULL;
    int kernels = 0;
    if (!vals || !own || !rem) {
        snprintf(err, errlen, "out of memory");
        goto done;
    }
    memcpy(rem, p->uses, (size_t)n * sizeof(int));

    size_t base = tensor_mem_live();
    tensor_mem_reset_peak();

    for (int k = 0; k < p->norder; k++) {
        int id = p->order[k];
        const Node *nd = &g->nodes[id];
        if (nd->op == OP_INPUT) {
            vals[id] = input;
            continue;
        }
        if (nd->op == OP_CONST) {
            vals[id] = nd->value;
            continue;
        }
        Tensor *out = tensor_new(nd->ndim, nd->shape);
        if (out == NULL) {
            snprintf(err, errlen, "out of memory running '%s'", nd->name);
            goto done;
        }
        const Tensor *ins[NODE_MAX_INPUTS] = {NULL, NULL, NULL};
        for (int j = 0; j < nd->ninputs; j++) ins[j] = vals[nd->inputs[j]];
        if (exec_node(nd, ins, out) != 0) {
            tensor_free(out);
            snprintf(err, errlen, "cannot execute node '%s' (%s)", nd->name, op_name(nd->op));
            goto done;
        }
        own[id] = out;
        vals[id] = out;
        kernels++;

        if (reuse) {
            /* Liveness: once the last consumer of a value has run, nothing will
             * read it again, so release it now instead of at the end. This lowers
             * peak memory from "sum of all intermediates" to "largest set alive at
             * the same time". An input read twice (ADD(x, x)) is counted twice in
             * uses, so it is freed only after both reads. */
            for (int j = 0; j < nd->ninputs; j++) {
                int src = nd->inputs[j];
                if (--rem[src] == 0 && own[src]) {
                    tensor_free(own[src]);
                    own[src] = NULL;
                    vals[src] = NULL;
                }
            }
            if (rem[id] == 0) { /* no consumers at all: dead code that was not removed */
                tensor_free(own[id]);
                own[id] = NULL;
                vals[id] = NULL;
            }
        }
    }

    /* Hand the output buffer to the caller. If the output is the INPUT or a
     * CONST (for example, after simplification), return a copy so the caller
     * always owns what it gets. */
    if (own[p->output_id]) {
        result = own[p->output_id];
        own[p->output_id] = NULL;
    } else {
        result = tensor_clone(vals[p->output_id]);
        if (result == NULL) snprintf(err, errlen, "out of memory copying output");
    }
    if (stats) {
        stats->peak_bytes = tensor_mem_peak() - base;
        stats->kernels_run = kernels;
    }
done:
    if (own)
        for (int i = 0; i < n; i++) tensor_free(own[i]);
    free(own);
    free(vals);
    free(rem);
    return result;
}

Tensor *graph_run(const Graph *g, const Tensor *input, int reuse, ExecStats *stats,
                  char *err, size_t errlen) {
    ExecPlan *p = plan_create(g, err, errlen);
    if (p == NULL) return NULL;
    Tensor *out = plan_run(p, g, input, reuse, stats, err, errlen);
    plan_free(p);
    return out;
}
