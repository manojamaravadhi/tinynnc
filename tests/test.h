#ifndef TINYNNC_TEST_H
#define TINYNNC_TEST_H

/*
 * A minimal test runner, so the project needs no external dependencies.
 *
 * A test is a `static void name(void)` function that uses the CHECK macros.
 * A failed CHECK records the failure with file:line and returns from the test
 * early. The remaining tests still run. run_test() prints PASS/FAIL per test,
 * and test_main.c returns a non-zero exit code if anything failed.
 */

#include <math.h>
#include <stdio.h>

extern int g_tests_run, g_tests_failed, g_current_failed;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            fprintf(stderr, "    %s:%d: CHECK failed: %s\n", __FILE__, __LINE__, \
                    #cond);                                                      \
            g_current_failed = 1;                                                \
            return;                                                              \
        }                                                                        \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                     \
    do {                                                                          \
        double a_ = (double)(a), b_ = (double)(b);                                \
        if (!(fabs(a_ - b_) <= (tol))) {                                          \
            fprintf(stderr, "    %s:%d: CHECK_NEAR failed: %s=%.9g vs %s=%.9g\n", \
                    __FILE__, __LINE__, #a, a_, #b, b_);                          \
            g_current_failed = 1;                                                 \
            return;                                                               \
        }                                                                         \
    } while (0)

/* Compare n floats elementwise. On failure, reports the first bad index. */
#define CHECK_ARRAY_NEAR(got, want, n, tol)                                         \
    do {                                                                            \
        for (size_t i_ = 0; i_ < (size_t)(n); i_++) {                               \
            double g_ = (got)[i_], w_ = (want)[i_];                                 \
            if (!(fabs(g_ - w_) <= (tol))) {                                        \
                fprintf(stderr, "    %s:%d: arrays differ at [%zu]: got %.9g want %.9g\n", \
                        __FILE__, __LINE__, i_, g_, w_);                            \
                g_current_failed = 1;                                               \
                return;                                                             \
            }                                                                       \
        }                                                                           \
    } while (0)

void run_test(const char *name, void (*fn)(void));
#define RUN(fn) run_test(#fn, fn)

/* One entry point per test file, called from test_main.c. */
void test_tensor(void);
void test_ops(void);
void test_graph(void);

#endif
