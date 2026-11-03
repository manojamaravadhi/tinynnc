#include "test.h"

int g_tests_run = 0, g_tests_failed = 0, g_current_failed = 0;

void run_test(const char *name, void (*fn)(void)) {
    g_current_failed = 0;
    fn();
    g_tests_run++;
    if (g_current_failed) g_tests_failed++;
    printf("  [%s] %s\n", g_current_failed ? "FAIL" : "PASS", name);
}

int main(void) {
    /* Line-buffer stdout so PASS/FAIL lines appear in order with the stderr
     * diagnostics, even when the output is piped (as in CI or `make test | tee`). */
    setvbuf(stdout, NULL, _IOLBF, 0);
    printf("== tensor ==\n");
    test_tensor();
    printf("== ops ==\n");
    test_ops();
    printf("== graph ==\n");
    test_graph();
    printf("== parser ==\n");
    test_parser();
    printf("== executor ==\n");
    test_exec();

    printf("\n%d tests, %d passed, %d failed\n", g_tests_run,
           g_tests_run - g_tests_failed, g_tests_failed);
    if (g_tests_failed) {
        printf("RESULT: FAIL\n");
        return 1;
    }
    printf("RESULT: OK\n");
    return 0;
}
