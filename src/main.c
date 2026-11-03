#define _POSIX_C_SOURCE 200809L /* clock_gettime */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "executor.h"
#include "graph.h"
#include "parser.h"
#include "passes.h"
#include "timer.h"

static void usage(FILE *out) {
    fprintf(out,
            "usage: tinynnc <model.txt> [options]\n"
            "  --input <spec>      input values: \"1,2,3,4\", a file of numbers, or random[:seed]\n"
            "                      (default: random:1)\n"
            "  --opt <spec>        none | all | comma list of fold,simplify,cse,dce,fuse\n"
            "                      (default: all)\n"
            "  --dump-ir           print the IR before and after optimization\n"
            "  --dot <file>        write the optimized graph as Graphviz .dot\n"
            "  --dot-before <file> write the unoptimized graph as Graphviz .dot\n"
            "  --verify            also run the unoptimized graph and compare outputs (tol 1e-5)\n"
            "  --no-reuse          keep all intermediates alive (disables liveness freeing)\n"
            "  --bench [N]         time N runs (default 200) of unoptimized vs optimized\n"
            "  --raw               print only the output values (for scripts)\n");
}

/* Parses numbers separated by commas/whitespace. Returns the count or -1 on a bad token. */
static long parse_numbers(const char *s, float *out, size_t cap) {
    long n = 0;
    while (*s) {
        while (*s == ',' || *s == ' ' || *s == '\t' || *s == '\n' || *s == '\r') s++;
        if (!*s) break;
        char *end;
        float v = strtof(s, &end);
        if (end == s) return -1;
        if ((size_t)n < cap) out[n] = v;
        n++;
        s = end;
    }
    return n;
}

static char *read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    size_t cap = 4096, len = 0;
    char *buf = malloc(cap);
    while (buf) {
        if (len + 1 >= cap) {
            char *nb = realloc(buf, cap * 2);
            if (!nb) {
                free(buf);
                buf = NULL;
                break;
            }
            buf = nb;
            cap *= 2;
        }
        size_t got = fread(buf + len, 1, cap - 1 - len, f);
        if (got == 0) break;
        len += got;
    }
    fclose(f);
    if (buf) buf[len] = '\0';
    return buf;
}

/* Builds the input tensor with the shape of the model's INPUT node. */
static Tensor *make_input(const Graph *g, const char *spec) {
    int id = graph_input_id(g);
    if (id < 0) return NULL;
    const Node *in = &g->nodes[id];
    Tensor *t = tensor_new(in->ndim, in->shape);
    if (!t) return NULL;
    if (strncmp(spec, "random", 6) == 0) {
        unsigned seed = spec[6] == ':' ? (unsigned)strtoul(spec + 7, NULL, 10) : 1u;
        tensor_fill_random(t, &seed, 1.0f);
        return t;
    }
    char *file = read_file(spec);
    const char *text = file ? file : spec; /* not a readable file: treat as a literal list */
    long n = parse_numbers(text, t->data, t->size);
    free(file);
    if (n != (long)t->size) {
        if (n < 0) fprintf(stderr, "error: --input contains a value that is not a number\n");
        else fprintf(stderr, "error: --input has %ld values, INPUT '%s' needs %zu\n", n,
                         in->name, t->size);
        tensor_free(t);
        return NULL;
    }
    return t;
}

static int write_dot(const Graph *g, const char *path, const char *title) {
    FILE *f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "error: cannot write %s\n", path);
        return -1;
    }
    graph_to_dot(g, f, title);
    fclose(f);
    return 0;
}

/* Times `iters` runs of a plan and reports mean and standard deviation. */
static void bench_plan(const char *label, const ExecPlan *p, const Graph *g, const Tensor *x,
                       int iters, int reuse) {
    char err[256];
    double sum = 0, sumsq = 0;
    for (int i = 0; i < 10; i++) tensor_free(plan_run(p, g, x, reuse, NULL, err, sizeof err));
    for (int i = 0; i < iters; i++) {
        double t0 = now_seconds();
        Tensor *y = plan_run(p, g, x, reuse, NULL, err, sizeof err);
        double dt = now_seconds() - t0;
        tensor_free(y);
        sum += dt;
        sumsq += dt * dt;
    }
    double mean = sum / iters;
    double var = sumsq / iters - mean * mean;
    printf("  %-12s %10.2f us  (std %.2f us, %d runs)\n", label, mean * 1e6,
           sqrt(var > 0 ? var : 0) * 1e6, iters);
}

int main(int argc, char **argv) {
    const char *model = NULL, *input_spec = "random:1", *opt = "all";
    const char *dot = NULL, *dot_before = NULL;
    int dump = 0, verify = 0, reuse = 1, bench = 0, raw = 0, rc = 1;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        int has_next = i + 1 < argc;
        if (!strcmp(a, "--input") && has_next) input_spec = argv[++i];
        else if (!strcmp(a, "--opt") && has_next) opt = argv[++i];
        else if (!strcmp(a, "--dot") && has_next) dot = argv[++i];
        else if (!strcmp(a, "--dot-before") && has_next) dot_before = argv[++i];
        else if (!strcmp(a, "--dump-ir")) dump = 1;
        else if (!strcmp(a, "--verify")) verify = 1;
        else if (!strcmp(a, "--no-reuse")) reuse = 0;
        else if (!strcmp(a, "--raw")) raw = 1;
        else if (!strcmp(a, "--bench")) {
            bench = 200;
            if (has_next && argv[i + 1][0] != '-') bench = atoi(argv[++i]);
            if (bench <= 0) bench = 200;
        } else if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
            usage(stdout);
            return 0;
        } else if (a[0] != '-' && model == NULL) model = a;
        else {
            fprintf(stderr, "error: unknown or incomplete option '%s'\n", a);
            usage(stderr);
            return 2;
        }
    }
    if (model == NULL) {
        usage(stderr);
        return 2;
    }

    char err[512];
    unsigned mask = 0;
    if (passes_parse(opt, &mask, err, sizeof err) != 0) {
        fprintf(stderr, "error: %s\n", err);
        return 2;
    }
    Graph *g = parse_model_file(model, err, sizeof err);
    if (g == NULL) {
        fprintf(stderr, "%s\n", err);
        return 2;
    }
    Graph *orig = graph_clone(g); /* unoptimized copy, for --verify and --bench */
    Tensor *x = make_input(g, input_spec);
    Tensor *y = NULL, *y_ref = NULL;
    ExecPlan *plan = NULL, *plan_ref = NULL;
    if (orig == NULL || x == NULL) goto out;

    if (dot_before && write_dot(g, dot_before, "before")) goto out;
    if (dump && !raw) {
        printf("== IR before optimization ==\n");
        graph_print(g, stdout);
    }
    if (run_passes(g, mask, NULL, raw ? NULL : stdout, err, sizeof err) != 0) {
        fprintf(stderr, "error: %s\n", err);
        goto out;
    }
    if (dump && !raw) {
        printf("== IR after optimization ==\n");
        graph_print(g, stdout);
    }
    if (dot && write_dot(g, dot, "after")) goto out;

    ExecStats st;
    plan = plan_create(g, err, sizeof err);
    if (plan) y = plan_run(plan, g, x, reuse, &st, err, sizeof err);
    if (y == NULL) {
        fprintf(stderr, "error: %s\n", err);
        goto out;
    }
    if (raw) {
        for (size_t i = 0; i < y->size; i++) printf(i ? " %.9g" : "%.9g", y->data[i]);
        printf("\n");
    } else {
        printf("== output ==\n");
        tensor_print(y, stdout);
        printf("kernels run: %d, peak activation memory: %zu bytes (buffer reuse %s)\n",
               st.kernels_run, st.peak_bytes, reuse ? "on" : "off");
    }
    rc = 0;

    if (verify || bench) {
        plan_ref = plan_create(orig, err, sizeof err);
        if (plan_ref == NULL) {
            fprintf(stderr, "error: %s\n", err);
            rc = 1;
            goto out;
        }
    }
    if (verify) {
        y_ref = plan_run(plan_ref, orig, x, reuse, NULL, err, sizeof err);
        if (y_ref == NULL || !tensor_same_shape(y, y_ref)) {
            fprintf(stderr, "verify: FAIL (%s)\n", y_ref ? "shape differs" : err);
            rc = 1;
            goto out;
        }
        double md = 0;
        for (size_t i = 0; i < y->size; i++) {
            double d = fabs((double)y->data[i] - (double)y_ref->data[i]);
            if (!(d <= md)) md = d; /* also propagates NaN differences */
        }
        int ok = md <= 1e-5;
        fprintf(raw ? stderr : stdout, "verify: max |optimized - unoptimized| = %.3g -> %s\n",
                md, ok ? "PASS" : "FAIL");
        if (!ok) rc = 1;
    }
    if (bench && !raw) {
        printf("== bench (%d runs each, after 10 warm-up runs) ==\n", bench);
        bench_plan("unoptimized", plan_ref, orig, x, bench, reuse);
        bench_plan("optimized", plan, g, x, bench, reuse);
    }
out:
    plan_free(plan);
    plan_free(plan_ref);
    tensor_free(x);
    tensor_free(y);
    tensor_free(y_ref);
    graph_free(g);
    graph_free(orig);
    return rc;
}
