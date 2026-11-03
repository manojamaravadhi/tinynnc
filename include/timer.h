#ifndef TINYNNC_TIMER_H
#define TINYNNC_TIMER_H

/* Monotonic wall clock in seconds. CLOCK_MONOTONIC never jumps when the system
 * time is changed, so it is the right clock for measuring intervals.
 * Files that include this header must define _POSIX_C_SOURCE (or _DARWIN_C_SOURCE)
 * before any system include. */
#include <time.h>

static inline double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

#endif
