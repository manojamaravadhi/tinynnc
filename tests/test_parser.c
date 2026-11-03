#include <stdlib.h>
#include <string.h>

#include "parser.h"
#include "test.h"

static const char *k_valid =
    "# comment line\n"
    "INPUT x 1 4\n"
    "CONST W1 4 3\n"
    "  1 0 0\n  0 1 0\n  0 0 1\n  1 1 1   # values may wrap lines\n"
    "CONST B1 1 3 0.5 -0.5 0\n"
    "MATMUL t1 x W1\n"
    "ADD t2 t1 B1\n"
    "RELU t3 t2\n"
    "OUTPUT t3\n";

static void parses_valid_model(void) {
    char err[512] = "";
    Graph *g = parse_model_string(k_valid, "valid", err, sizeof err);
    CHECK(g != NULL);
    CHECK(g->count == 6);
    int t3 = graph_find(g, "t3");
    CHECK(t3 >= 0 && g->output == t3);
    CHECK(g->nodes[t3].shape[0] == 1 && g->nodes[t3].shape[1] == 3);
    CHECK(g->nodes[graph_find(g, "W1")].value->data[9] == 1.0f);
    CHECK(g->nodes[graph_find(g, "B1")].value->data[1] == -0.5f);
    CHECK(g->nodes[t3].line == 11);
    graph_free(g);
}

/* Each malformed model must fail with an error at the right line and with a
 * message that contains the expected text. */
typedef struct {
    const char *text;
    const char *expect; /* substring of the error, including the line number */
} BadCase;

static const BadCase k_bad[] = {
    {"", "<t>:1: error: model has no INPUT"},
    {"INPUT x 1 4\n", "model has no OUTPUT"},
    {"INPUT x 1 4\nFOO y x\nOUTPUT x\n", ":2: error: unknown statement 'FOO'"},
    {"INPUT x 1 4\nRELU y z\nOUTPUT y\n", ":2: error: RELU: undefined value 'z'"},
    {"INPUT x 1 4\nRELU x x\n", ":2: error: 'x' is already defined (line 1)"},
    {"INPUT x 1 4\nRELU y\n", ":2: error: RELU: missing operand 1"},
    {"INPUT x 1 4\nRELU y x x\n", ":2: error: RELU: unexpected extra token 'x'"},
    {"INPUT x 1 4\nCONST W 2 2 1 2 3\n", ":2: error: CONST 'W': expected 4 values, found only 3"},
    {"INPUT x 1 4\nCONST W 2 2 1 2 abc 4\n", ":2: error: expected a number for CONST 'W', got 'abc'"},
    {"INPUT x 1 4\nCONST W 2\n", ":2: error: CONST 'W': expected 2 dimensions"},
    {"INPUT x 0 4\n", ":1: error: expected a positive dimension, got '0'"},
    {"INPUT x -1 4\n", ":1: error: expected a positive dimension, got '-1'"},
    {"INPUT x 1 2 3 4 5\n", ":1: error: INPUT 'x': at most 4 dimensions"},
    {"INPUT x 99999999999 4\n", ":1: error: expected a positive dimension"},
    {"INPUT x 16777216 16777216\n", ":1: error: INPUT 'x': tensor too large"},
    {"INPUT x\n", ":1: error: INPUT 'x': missing shape"},
    {"INPUT x 1 4\nINPUT y 1 4\n", ":2: error: only one INPUT is supported"},
    {"INPUT 1x 1 4\n", ":1: error: invalid name '1x'"},
    {"INPUT abcdefghijabcdefghijabcdefghij_toolong 1 4\n", "is too long"},
    {"INPUT x 1 4\nOUTPUT x\nOUTPUT x\n", ":3: error: OUTPUT given more than once"},
    {"INPUT x 1 4\nCONST W 3 3 1 2 3 4 5 6 7 8 9\nMATMUL y x W\nOUTPUT y\n",
     ":3: error: MATMUL 'y': inner dimensions differ: 'x' is [1x4], 'W' is [3x3] (4 != 3)"},
    {"INPUT x 2 3\nCONST b 1 4 1 2 3 4\nADD y x b\nOUTPUT y\n",
     ":3: error: ADD 'y': operand 'b' has shape [1x4], expected [2x3] or [1x3]"},
    {"INPUT x 1 4\nMATMULXYZW_LONG y x x\n", "unknown statement 'MATMULXYZW_LONG'"},
};

static void rejects_malformed_models(void) {
    for (size_t i = 0; i < sizeof k_bad / sizeof k_bad[0]; i++) {
        char err[512] = "";
        Graph *g = parse_model_string(k_bad[i].text, "<t>", err, sizeof err);
        if (g != NULL || strstr(err, k_bad[i].expect) == NULL) {
            fprintf(stderr, "    case %zu: expected error containing \"%s\"\n      got: %s\n", i,
                    k_bad[i].expect, g ? "(parsed OK)" : err);
            graph_free(g);
            CHECK(0);
        }
    }
}

static void missing_file_is_error(void) {
    char err[256] = "";
    CHECK(parse_model_file("/nonexistent/model.txt", err, sizeof err) == NULL);
    CHECK(strstr(err, "cannot open") != NULL);
}

/* Robustness: take a valid model and randomly corrupt it 3000 times (flip bytes,
 * truncate, insert junk). The parser must return either a graph or an error. It
 * must never crash. Under `make debug` this also proves there are no out-of-bounds
 * reads, use-after-free, or UB in the error paths. */
static void fuzz_never_crashes(void) {
    const char junk[] = "x#\n 0-9.eE+-ABCDEFZ_ \t";
    size_t len = strlen(k_valid);
    char *buf = malloc(len * 2 + 8);
    CHECK(buf != NULL);
    unsigned seed = 7;
    int parsed = 0, rejected = 0;
    for (int iter = 0; iter < 3000; iter++) {
        memcpy(buf, k_valid, len + 1);
        size_t cur = len;
        int edits = 1 + iter % 4;
        for (int e = 0; e < edits; e++) {
            seed = seed * 1103515245u + 12345u;
            size_t at = (seed >> 8) % (cur ? cur : 1);
            int kind = (seed >> 20) % 3;
            char c = junk[(seed >> 4) % (sizeof junk - 1)];
            if (kind == 0) buf[at] = c;             /* overwrite */
            else if (kind == 1) buf[cur = at] = 0;  /* truncate */
            else if (cur < len * 2) {               /* insert */
                memmove(buf + at + 1, buf + at, cur - at + 1);
                buf[at] = c;
                cur++;
            }
        }
        char err[512];
        Graph *g = parse_model_string(buf, "fuzz", err, sizeof err);
        if (g) parsed++;
        else rejected++;
        graph_free(g);
    }
    free(buf);
    CHECK(parsed + rejected == 3000);
    CHECK(rejected > 0);
}

void test_parser(void) {
    RUN(parses_valid_model);
    RUN(rejects_malformed_models);
    RUN(missing_file_is_error);
    RUN(fuzz_never_crashes);
}
