#include "parser.h"

#include <ctype.h>
#include <errno.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

/*
 * The lexer splits the text into whitespace-separated tokens and records each
 * token's line. Statements are line-oriented ("RELU t3 t2" must fit on one line),
 * except CONST values, which may continue over the following lines. That way
 * large weight matrices can be wrapped.
 */
typedef struct {
    const char *p;   /* current position */
    int line;        /* line of *p */
} Lexer;

typedef struct {
    const char *s;
    size_t len;
    int line;
} Token;

typedef struct {
    Lexer lx;
    const char *src;
    char *err;
    size_t errlen;
    Graph *g;
} Parser;

static void skip_space(Lexer *lx) {
    for (;;) {
        char c = *lx->p;
        if (c == '\n') {
            lx->line++;
            lx->p++;
        } else if (c == ' ' || c == '\t' || c == '\r' || c == '\f' || c == '\v') {
            lx->p++;
        } else if (c == '#') { /* comment runs to end of line */
            while (*lx->p && *lx->p != '\n') lx->p++;
        } else {
            return;
        }
    }
}

/* Looks at the next token without consuming it. Returns 0 at end of input. */
static int peek(Lexer *lx, Token *t) {
    skip_space(lx);
    if (*lx->p == '\0') return 0;
    const char *s = lx->p;
    while (*s && !isspace((unsigned char)*s) && *s != '#') s++;
    t->s = lx->p;
    t->len = (size_t)(s - lx->p);
    t->line = lx->line;
    return 1;
}

static void consume(Lexer *lx, const Token *t) { lx->p = t->s + t->len; }

/* Next token, but only if it is on `line`. Used for line-oriented statements. */
static int next_on_line(Parser *ps, int line, Token *t) {
    if (!peek(&ps->lx, t) || t->line != line) return 0;
    consume(&ps->lx, t);
    return 1;
}

static void fail(Parser *ps, int line, const char *fmt, ...) {
    char msg[384];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    snprintf(ps->err, ps->errlen, "%s:%d: error: %s", ps->src, line, msg);
}

/* Copies a token into buf for printing in error messages (truncated, NUL-terminated). */
static const char *tokstr(const Token *t, char *buf, size_t len) {
    size_t n = t->len < len - 1 ? t->len : len - 1;
    memcpy(buf, t->s, n);
    buf[n] = '\0';
    if (t->len > n && len > 4) memcpy(buf + len - 4, "...", 4);
    return buf;
}

static int tok_is(const Token *t, const char *word) {
    return t->len == strlen(word) && memcmp(t->s, word, t->len) == 0;
}

/* Identifiers: [A-Za-z_][A-Za-z0-9_]*, shorter than NODE_NAME_LEN. */
static int parse_name(Parser *ps, const Token *t, char *out) {
    char b[48];
    if (t->len == 0 || t->len >= NODE_NAME_LEN) {
        fail(ps, t->line, "name '%s' is too long (max %d chars)", tokstr(t, b, sizeof b),
             NODE_NAME_LEN - 1);
        return -1;
    }
    for (size_t i = 0; i < t->len; i++) {
        unsigned char c = (unsigned char)t->s[i];
        if (!(isalpha(c) || c == '_' || (i > 0 && isdigit(c)))) {
            fail(ps, t->line, "invalid name '%s' (use letters, digits, _)",
                 tokstr(t, b, sizeof b));
            return -1;
        }
    }
    memcpy(out, t->s, t->len);
    out[t->len] = '\0';
    return 0;
}

static int parse_dim(Parser *ps, const Token *t, int *out) {
    char b[48];
    if (t->len == 0 || t->len > 9) goto bad; /* also catches overflow of int */
    for (size_t i = 0; i < t->len; i++)
        if (!isdigit((unsigned char)t->s[i])) goto bad;
    long v = strtol(tokstr(t, b, sizeof b), NULL, 10);
    if (v < 1 || v > PARSER_MAX_ELEMS) goto bad;
    *out = (int)v;
    return 0;
bad:
    fail(ps, t->line, "expected a positive dimension, got '%s'", tokstr(t, b, sizeof b));
    return -1;
}

static int parse_float_tok(Parser *ps, const Token *t, float *out, const char *what) {
    char b[64];
    if (t->len >= sizeof b) goto bad;
    tokstr(t, b, sizeof b);
    char *end = NULL;
    errno = 0;
    float v = strtof(b, &end);
    if (end == b || *end != '\0') goto bad;
    *out = v;
    return 0;
bad:
    fail(ps, t->line, "expected a number for %s, got '%s'", what, tokstr(t, b, 48));
    return -1;
}

/* Reads a name that must not already be defined. */
static int parse_new_name(Parser *ps, int line, const char *kw, char *name) {
    Token t;
    if (!next_on_line(ps, line, &t)) {
        fail(ps, line, "%s: missing name", kw);
        return -1;
    }
    if (parse_name(ps, &t, name)) return -1;
    if (graph_find(ps->g, name) >= 0) {
        int prev = ps->g->nodes[graph_find(ps->g, name)].line;
        fail(ps, line, "'%s' is already defined (line %d)", name, prev);
        return -1;
    }
    return 0;
}

/* Reads a name that must refer to an existing value. Returns its node id. */
static int parse_ref(Parser *ps, int line, const char *kw, int argno) {
    Token t;
    char name[NODE_NAME_LEN];
    if (!next_on_line(ps, line, &t)) {
        fail(ps, line, "%s: missing operand %d", kw, argno);
        return -1;
    }
    if (parse_name(ps, &t, name)) return -1;
    int id = graph_find(ps->g, name);
    if (id < 0) fail(ps, line, "%s: undefined value '%s'", kw, name);
    return id;
}

static int expect_eol(Parser *ps, int line, const char *kw) {
    Token t;
    char b[48];
    if (next_on_line(ps, line, &t)) {
        fail(ps, line, "%s: unexpected extra token '%s'", kw, tokstr(&t, b, sizeof b));
        return -1;
    }
    return 0;
}

/* Infers the new node's shape right away, so shape errors point at its line. */
static int finish_node(Parser *ps, int id, int line) {
    char msg[320];
    ps->g->nodes[id].line = line;
    if (graph_infer_node(ps->g, id, msg, sizeof msg) != 0) {
        fail(ps, line, "%s", msg);
        return -1;
    }
    return 0;
}

static int stmt_input(Parser *ps, int line) {
    char name[NODE_NAME_LEN];
    int shape[TENSOR_MAX_DIMS], nd = 0;
    Token t;
    if (graph_input_id(ps->g) >= 0) {
        fail(ps, line, "only one INPUT is supported");
        return -1;
    }
    if (parse_new_name(ps, line, "INPUT", name)) return -1;
    while (next_on_line(ps, line, &t)) {
        if (nd == TENSOR_MAX_DIMS) {
            fail(ps, line, "INPUT '%s': at most %d dimensions", name, TENSOR_MAX_DIMS);
            return -1;
        }
        if (parse_dim(ps, &t, &shape[nd++])) return -1;
    }
    if (nd == 0) {
        fail(ps, line, "INPUT '%s': missing shape", name);
        return -1;
    }
    size_t numel = shape_numel(nd, shape);
    if (numel == 0 || numel > (size_t)PARSER_MAX_ELEMS) {
        fail(ps, line, "INPUT '%s': tensor too large", name);
        return -1;
    }
    int id = graph_add_input(ps->g, name, nd, shape);
    if (id < 0) {
        fail(ps, line, "cannot add INPUT '%s'", name);
        return -1;
    }
    return finish_node(ps, id, line);
}

static int stmt_const(Parser *ps, int line) {
    char name[NODE_NAME_LEN], what[64];
    int shape[2];
    Token t;
    if (parse_new_name(ps, line, "CONST", name)) return -1;
    for (int i = 0; i < 2; i++) {
        if (!next_on_line(ps, line, &t)) {
            fail(ps, line, "CONST '%s': expected 2 dimensions (rows cols)", name);
            return -1;
        }
        if (parse_dim(ps, &t, &shape[i])) return -1;
    }
    size_t numel = (size_t)shape[0] * (size_t)shape[1];
    if (numel > (size_t)PARSER_MAX_ELEMS) {
        fail(ps, line, "CONST '%s': tensor too large (%zu elements)", name, numel);
        return -1;
    }
    Tensor *v = tensor_new(2, shape);
    if (v == NULL) {
        fail(ps, line, "CONST '%s': out of memory", name);
        return -1;
    }
    snprintf(what, sizeof what, "CONST '%s'", name);
    for (size_t i = 0; i < numel; i++) {
        if (!peek(&ps->lx, &t)) {
            fail(ps, line, "CONST '%s': expected %zu values, found only %zu before end of file",
                 name, numel, i);
            tensor_free(v);
            return -1;
        }
        consume(&ps->lx, &t);
        if (parse_float_tok(ps, &t, &v->data[i], what)) {
            tensor_free(v);
            return -1;
        }
    }
    int id = graph_add_const(ps->g, name, v); /* takes ownership of v */
    if (id < 0) {
        fail(ps, line, "cannot add CONST '%s'", name);
        return -1;
    }
    return finish_node(ps, id, line);
}

static int stmt_op(Parser *ps, int line, OpType op) {
    char name[NODE_NAME_LEN];
    int inputs[NODE_MAX_INPUTS];
    const char *kw = op_name(op);
    if (parse_new_name(ps, line, kw, name)) return -1;
    for (int i = 0; i < op_arity(op); i++)
        if ((inputs[i] = parse_ref(ps, line, kw, i + 1)) < 0) return -1;
    if (expect_eol(ps, line, kw)) return -1;
    int id = graph_add_op(ps->g, op, name, inputs, op_arity(op));
    if (id < 0) {
        fail(ps, line, "cannot add %s '%s'", kw, name);
        return -1;
    }
    return finish_node(ps, id, line);
}

static int stmt_output(Parser *ps, int line) {
    if (ps->g->output >= 0) {
        fail(ps, line, "OUTPUT given more than once");
        return -1;
    }
    int id = parse_ref(ps, line, "OUTPUT", 1);
    if (id < 0 || expect_eol(ps, line, "OUTPUT")) return -1;
    ps->g->output = id;
    return 0;
}

Graph *parse_model_string(const char *text, const char *srcname, char *err, size_t errlen) {
    Parser ps = {{text, 1}, srcname ? srcname : "<string>", err, errlen, graph_new()};
    Token t;
    int last_line = 1;
    if (ps.g == NULL) {
        snprintf(err, errlen, "out of memory");
        return NULL;
    }
    while (peek(&ps.lx, &t)) {
        consume(&ps.lx, &t);
        int line = last_line = t.line, rc;
        char kw[16], b[48];
        OpType op;
        if (t.len >= sizeof kw) {
            fail(&ps, line, "unknown statement '%s'", tokstr(&t, b, sizeof b));
            goto error;
        }
        memcpy(kw, t.s, t.len);
        kw[t.len] = '\0';
        if (tok_is(&t, "INPUT")) rc = stmt_input(&ps, line);
        else if (tok_is(&t, "CONST")) rc = stmt_const(&ps, line);
        else if (tok_is(&t, "OUTPUT")) rc = stmt_output(&ps, line);
        else if (op_from_name(kw, &op) && op != OP_INPUT && op != OP_CONST)
            rc = stmt_op(&ps, line, op);
        else {
            fail(&ps, line, "unknown statement '%s'", kw);
            rc = -1;
        }
        if (rc) goto error;
    }
    if (graph_input_id(ps.g) < 0) {
        fail(&ps, last_line, "model has no INPUT");
        goto error;
    }
    if (ps.g->output < 0) {
        fail(&ps, last_line, "model has no OUTPUT");
        goto error;
    }
    return ps.g;
error:
    graph_free(ps.g);
    return NULL;
}

Graph *parse_model_file(const char *path, char *err, size_t errlen) {
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        snprintf(err, errlen, "%s: cannot open file", path);
        return NULL;
    }
    /* Read the whole file into memory. Model files are small, and a single
     * buffer keeps the lexer simple (it can peek ahead freely). */
    size_t cap = 1 << 16, len = 0;
    char *buf = malloc(cap);
    while (buf) {
        if (len + 1 >= cap) {
            if (cap >= (size_t)1 << 30) {
                snprintf(err, errlen, "%s: file too large", path);
                free(buf);
                fclose(f);
                return NULL;
            }
            char *nb = realloc(buf, cap * 2);
            if (nb == NULL) {
                free(buf);
                buf = NULL;
                break;
            }
            buf = nb;
            cap *= 2;
        }
        size_t got = fread(buf + len, 1, cap - 1 - len, f);
        len += got;
        if (got == 0) break;
    }
    fclose(f);
    if (buf == NULL) {
        snprintf(err, errlen, "%s: out of memory reading file", path);
        return NULL;
    }
    buf[len] = '\0';
    if (strlen(buf) != len) {
        snprintf(err, errlen, "%s: file contains a NUL byte (not a text model file)", path);
        free(buf);
        return NULL;
    }
    Graph *g = parse_model_string(buf, path, err, errlen);
    free(buf);
    return g;
}
