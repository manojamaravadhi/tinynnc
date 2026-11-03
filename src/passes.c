#include "passes.h"

#include <string.h>

/* Milestone M3 stub: only "none" is supported. The passes arrive in M4/M5. */

int passes_parse(const char *spec, unsigned *mask, char *err, size_t errlen) {
    if (strcmp(spec, "none") == 0) {
        *mask = 0;
        return 0;
    }
    snprintf(err, errlen, "unknown --opt value '%s' (passes not implemented yet)", spec);
    return -1;
}

int run_passes(Graph *g, unsigned mask, PassReport *report, FILE *log, char *err,
               size_t errlen) {
    (void)g;
    (void)log;
    if (report) memset(report, 0, sizeof *report);
    if (mask != 0) {
        snprintf(err, errlen, "passes not implemented yet");
        return -1;
    }
    return 0;
}
