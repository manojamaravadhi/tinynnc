/*
 * TinyNNC benchmark: measures each optimization pass on its own and all passes
 * together, on clean MLPs and on MLPs that contain exporter-style redundant ops.
 * It writes a markdown report (default docs/RESULTS.md). Every number in the
 * report is measured on this run. Nothing is hard-coded.
 *
 * Method: for each (model, pass set) we optimize a fresh copy of the graph, build
 * an execution plan once, do warm-up runs, then time many individual runs
 * with CLOCK_MONOTONIC. We report the mean and standard deviation. Each run includes
 * allocating and freeing intermediates, because that is part of the runtime's cost.
 */
#if defined(__APPLE__)
#define _DARWIN_C_SOURCE /* clock_gettime + sysctlbyname */
#else
#define _POSIX_C_SOURCE 200809L
#endif

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/utsname.h>
#include <time.h>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#include "executor.h"
#include "graph.h"
#include "passes.h"
#include "timer.h"

#ifndef TNN_CFLAGS
#define TNN_CFLAGS "(unknown)"
#endif

static unsigned g_seed = 42u;

static int add_rand_const(Graph *g, const char *name, int r, int c, float scale) {
    int s[2] = {r, c};
    Tensor *t = tensor_new(2, s);
    if (t) tensor_fill_random(t, &g_seed, scale);
    return graph_add_const(g, name, t);
}

static int add_fill_const(Graph *g, const char *name, int r, int c, int identity) {
    int s[2] = {r, c};
    Tensor *t = tensor_new(2, s);
    if (t && identity)
        for (int i = 0; i < r; i++) t->data[(size_t)i * c + i] = 1.0f;
    return graph_add_const(g, name, t);
}

static int op2(Graph *g, OpType op, const char *name, int a, int b) {
    int in[2] = {a, b};
    return graph_add_op(g, op, name, in, 2);
}

/* in -> hidden -> out MLP with ReLU on the hidden layer.
 * redundant = 1 adds what a naive exporter might emit: the bias written as a sum
 * of two constants, a doubled ReLU, a multiply by an identity matrix, an add of
 * zero, and a duplicated MATMUL feeding an unused debug output. */
static Graph *build_mlp(int batch, int in, int hidden, int out, int redundant) {
    Graph *g = graph_new();
    int s[2] = {batch, in};
    int x = graph_add_input(g, "x", 2, s);
    int dims[3] = {in, hidden, out};
    int cur = x;
    for (int l = 0; l < 2; l++) {
        int k = dims[l], n = dims[l + 1];
        int W = add_rand_const(g, "W", k, n, 1.0f / sqrtf((float)k));
        int b;
        if (redundant) {
            b = op2(g, OP_ADD, "b", add_rand_const(g, "ba", 1, n, 0.1f),
                    add_rand_const(g, "bb", 1, n, 0.1f));
        } else {
            b = add_rand_const(g, "b", 1, n, 0.1f);
        }
        int m = op2(g, OP_MATMUL, "mm", cur, W);
        if (redundant) {
            int dup = op2(g, OP_MATMUL, "mm_dup", cur, W);
            graph_add_op(g, OP_SIGMOID, "debug", &dup, 1);
        }
        cur = op2(g, OP_ADD, "add", m, b);
        if (l == 0) {
            cur = graph_add_op(g, OP_RELU, "relu", &cur, 1);
            if (redundant) {
                cur = graph_add_op(g, OP_RELU, "relu2", &cur, 1);
                cur = op2(g, OP_MATMUL, "ident", cur, add_fill_const(g, "I", n, n, 1));
                cur = op2(g, OP_ADD, "zero", cur, add_fill_const(g, "Z", 1, n, 0));
            }
        }
    }
    g->output = cur;
    char err[256];
    if (graph_infer_shapes(g, err, sizeof err) != 0) {
        fprintf(stderr, "bench: bad model: %s\n", err);
        exit(1);
    }
    return g;
}

typedef struct {
    const char *name;
    unsigned mask;
} PassSet;

static const PassSet k_sets[] = {
    {"none", 0},          {"fold", PASS_FOLD}, {"simplify", PASS_SIMPLIFY},
    {"cse", PASS_CSE},    {"dce", PASS_DCE},   {"fuse", PASS_FUSE},
    {"all", PASS_ALL},
};
#define NSETS ((int)(sizeof k_sets / sizeof k_sets[0]))

typedef struct {
    int ops;
    double flops;
    size_t peak_reuse, peak_noreuse;
    double mean_us, std_us;
    int iters;
    double maxdiff;
    /* timing state */
    Graph *g;
    ExecPlan *plan;
    int per_round;
    double sum, sumsq;
} Result;

/* Optimizes a copy of the model with `mask` and records the static numbers (ops,
 * FLOPs, peak memory, output difference). Timing happens later, in rounds. */
static void prepare(const Graph *base, unsigned mask, const Tensor *x, const Tensor *ref,
                    Result *res) {
    char err[256];
    memset(res, 0, sizeof *res);
    res->g = graph_clone(base);
    if (!res->g || run_passes(res->g, mask, NULL, NULL, err, sizeof err) != 0) {
        fprintf(stderr, "bench: passes failed: %s\n", err);
        exit(1);
    }
    GraphStats st;
    graph_stats(res->g, &st);
    res->ops = st.compute_nodes;
    res->flops = st.flops;
    res->plan = plan_create(res->g, err, sizeof err);
    ExecStats es;
    Tensor *y = plan_run(res->plan, res->g, x, 0, &es, err, sizeof err);
    res->peak_noreuse = es.peak_bytes;
    tensor_free(y);
    y = plan_run(res->plan, res->g, x, 1, &es, err, sizeof err);
    res->peak_reuse = es.peak_bytes;
    for (size_t i = 0; i < y->size; i++) {
        double d = fabs((double)y->data[i] - (double)ref->data[i]);
        if (!(d <= res->maxdiff)) res->maxdiff = d;
    }
    tensor_free(y);
    /* Warm up, and estimate the cost of one run to size each timing round at ~0.1 s. */
    double t0 = now_seconds();
    for (int i = 0; i < 20; i++)
        tensor_free(plan_run(res->plan, res->g, x, 1, NULL, err, sizeof err));
    double per = (now_seconds() - t0) / 20;
    res->per_round = (int)(0.1 / (per > 1e-9 ? per : 1e-9));
    if (res->per_round < 10) res->per_round = 10;
    if (res->per_round > 5000) res->per_round = 5000;
}

static void time_round(Result *r, const Tensor *x) {
    char err[256];
    for (int i = 0; i < r->per_round; i++) {
        double a = now_seconds();
        Tensor *out = plan_run(r->plan, r->g, x, 1, NULL, err, sizeof err);
        double dt = now_seconds() - a;
        tensor_free(out);
        r->sum += dt;
        r->sumsq += dt * dt;
        r->iters++;
    }
}

static void finish(Result *r) {
    double mean = r->sum / r->iters, var = r->sumsq / r->iters - mean * mean;
    r->mean_us = mean * 1e6;
    r->std_us = sqrt(var > 0 ? var : 0) * 1e6;
    plan_free(r->plan);
    graph_free(r->g);
    r->plan = NULL;
    r->g = NULL;
}

static void machine_info(FILE *f) {
    struct utsname u;
    char cpu[256] = "unknown";
    long long mem = 0;
    int ncpu = 0;
#if defined(__APPLE__)
    size_t len = sizeof cpu;
    sysctlbyname("machdep.cpu.brand_string", cpu, &len, NULL, 0);
    len = sizeof mem;
    sysctlbyname("hw.memsize", &mem, &len, NULL, 0);
    len = sizeof ncpu;
    sysctlbyname("hw.ncpu", &ncpu, &len, NULL, 0);
#else
    FILE *ci = fopen("/proc/cpuinfo", "r");
    char line[512];
    while (ci && fgets(line, sizeof line, ci)) {
        if (strncmp(line, "model name", 10) == 0) {
            char *c = strchr(line, ':');
            if (c) snprintf(cpu, sizeof cpu, "%s", c + 2);
            cpu[strcspn(cpu, "\n")] = 0;
        }
        if (strncmp(line, "processor", 9) == 0) ncpu++;
    }
    if (ci) fclose(ci);
#endif
    uname(&u);
    fprintf(f, "- **CPU:** %s (%d logical cores; the benchmark is single-threaded)\n", cpu, ncpu);
    if (mem) fprintf(f, "- **RAM:** %.0f GB\n", (double)mem / (1024.0 * 1024 * 1024));
    fprintf(f, "- **OS:** %s %s (%s)\n", u.sysname, u.release, u.machine);
#if defined(__VERSION__)
    fprintf(f, "- **Compiler:** %s\n", __VERSION__);
#endif
    fprintf(f, "- **Flags:** `%s`\n", TNN_CFLAGS);
}

typedef struct {
    const char *label;
    int batch, in, hidden, out, redundant;
} ModelCfg;

static const ModelCfg k_models[] = {
    {"MLP 64x128x64, batch 1, clean", 1, 64, 128, 64, 0},
    {"MLP 64x128x64, batch 32, clean", 32, 64, 128, 64, 0},
    {"MLP 256x512x256, batch 32, clean", 32, 256, 512, 256, 0},
    {"MLP 64x128x64, batch 32, with redundant ops", 32, 64, 128, 64, 1},
    {"MLP 256x512x256, batch 32, with redundant ops", 32, 256, 512, 256, 1},
};
#define NMODELS ((int)(sizeof k_models / sizeof k_models[0]))

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "docs/RESULTS.md";
    FILE *f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "bench: cannot write %s\n", path);
        return 1;
    }
    char date[64];
    time_t now = time(NULL);
    strftime(date, sizeof date, "%Y-%m-%d %H:%M UTC", gmtime(&now));

    fprintf(f, "# TinyNNC benchmark results\n\n");
    fprintf(f, "Generated by `make bench` (`bench/bench.c`) on %s. Every number below was "
               "measured during that run. None are typed in by hand.\n\n", date);
    fprintf(f, "## Machine\n\n");
    machine_info(f);
    fprintf(f, "\n## Method\n\n"
               "- Each pass set is applied to a fresh copy of the model, and an execution plan "
               "is built once. After 20 warm-up runs, the pass sets are timed in 5 interleaved "
               "rounds (none, fold, ..., all, then again). Each round is about 0.1 s per pass set, so "
               "slow drift (CPU frequency, background load) affects all of them about equally. "
               "Mean and standard deviation are over all individual runs.\n"
               "- Time covers one full forward pass, including allocating and freeing "
               "intermediate buffers.\n"
               "- **Peak mem** = peak bytes of activations (intermediates + output) during one "
               "run, with liveness-based freeing on. Weights and the input are not included.\n"
               "- **FLOPs** are estimates (2 per multiply-add, 1 per elementwise op).\n"
               "- **max |diff|** = largest absolute difference from the unoptimized output on "
               "the same input.\n"
               "- Weights are deterministic pseudo-random values (seed 42), and inputs are "
               "uniform in [-1, 1).\n\n");

    Result summary[NMODELS][NSETS];
    for (int mi = 0; mi < NMODELS; mi++) {
        const ModelCfg *mc = &k_models[mi];
        Graph *g = build_mlp(mc->batch, mc->in, mc->hidden, mc->out, mc->redundant);
        int s[2] = {mc->batch, mc->in};
        Tensor *x = tensor_new(2, s);
        tensor_fill_random(x, &g_seed, 1.0f);
        char err[256];
        Tensor *ref = graph_run(g, x, 1, NULL, err, sizeof err);

        fprintf(f, "## %s\n\n", mc->label);
        fprintf(f, "| passes | compute ops | est. FLOPs | peak mem (bytes) | time (us), mean ± std "
                   "| runs | speedup vs none | max \\|diff\\| |\n");
        fprintf(f, "|---|---:|---:|---:|---:|---:|---:|---:|\n");
        printf("%s\n", mc->label);
        /* Time the pass sets in interleaved rounds (none, fold, ..., all, none,
         * fold, ...). Slow drift such as CPU frequency changes or other processes
         * then affects every configuration about equally, instead of favouring
         * whichever one happens to run last. */
        const int rounds = 5;
        for (int si = 0; si < NSETS; si++) prepare(g, k_sets[si].mask, x, ref, &summary[mi][si]);
        for (int rd = 0; rd < rounds; rd++)
            for (int si = 0; si < NSETS; si++) time_round(&summary[mi][si], x);
        for (int si = 0; si < NSETS; si++) finish(&summary[mi][si]);
        for (int si = 0; si < NSETS; si++) {
            Result *r = &summary[mi][si];
            double speedup = summary[mi][0].mean_us / r->mean_us;
            fprintf(f, "| %s | %d | %.0f | %zu | %.2f ± %.2f | %d | %.2fx | %.1e |\n",
                    k_sets[si].name, r->ops, r->flops, r->peak_reuse, r->mean_us, r->std_us,
                    r->iters, speedup, r->maxdiff);
            printf("  %-9s ops=%-3d peak=%-8zu %10.2f us ± %-8.2f speedup %.2fx  diff %.1e\n",
                   k_sets[si].name, r->ops, r->peak_reuse, r->mean_us, r->std_us, speedup,
                   r->maxdiff);
        }
        fprintf(f, "\n");
        tensor_free(ref);
        tensor_free(x);
        graph_free(g);
    }

    fprintf(f, "## Liveness-based buffer freeing: peak activation memory (bytes)\n\n");
    fprintf(f, "| model | none, keep all | none, free after last use | all passes, keep all "
               "| all passes, free after last use |\n|---|---:|---:|---:|---:|\n");
    for (int mi = 0; mi < NMODELS; mi++) {
        const Result *a = &summary[mi][0], *b = &summary[mi][NSETS - 1];
        fprintf(f, "| %s | %zu | %zu | %zu | %zu |\n", k_models[mi].label, a->peak_noreuse,
                a->peak_reuse, b->peak_noreuse, b->peak_reuse);
    }
    fclose(f);
    printf("wrote %s\n", path);
    return 0;
}
