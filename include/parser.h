#ifndef TINYNNC_PARSER_H
#define TINYNNC_PARSER_H

/*
 * Model file parser. The grammar is documented in docs/DESIGN.md.
 *
 * Every error is reported as "<source>:<line>: error: <message>" in err, and the
 * function returns NULL. The parser treats its input as untrusted: it never
 * crashes, never reads past the buffer, and bounds every size before allocating.
 * Shapes are checked statement by statement, so shape errors also carry a line number.
 */

#include "graph.h"

/* Upper bound on the elements of one INPUT/CONST tensor (64 MB of floats). This
 * stops a one-line file from requesting a multi-gigabyte allocation. */
#define PARSER_MAX_ELEMS (1L << 24)

Graph *parse_model_string(const char *text, const char *srcname, char *err, size_t errlen);
Graph *parse_model_file(const char *path, char *err, size_t errlen);

#endif
