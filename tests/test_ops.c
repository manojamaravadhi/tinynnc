#include <string.h>

#include "ops.h"
#include "test.h"

#define TOL 1e-6

/* A = [[1,2,3],[4,5,6]] (2x3), B = [[7,8],[9,10],[11,12]] (3x2) */
static const float A23[6] = {1, 2, 3, 4, 5, 6};
static const float B32[6] = {7, 8, 9, 10, 11, 12};

static void matmul_hand_computed(void) {
    /* Row 0: [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
     * Row 1: [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154] */
    float C[4];
    op_matmul(A23, B32, C, 2, 3, 2);
    const float want[4] = {58, 64, 139, 154};
    CHECK_ARRAY_NEAR(C, want, 4, TOL);
}

static void matmul_identity(void) {
    const float I[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    float C[6];
    op_matmul(A23, I, C, 2, 3, 3);
    CHECK_ARRAY_NEAR(C, A23, 6, 0.0); /* multiplying by I must be exact */
}

static void matmul_vector(void) {
    /* (1x3) * (3x1): 1*4 + 2*5 + 3*6 = 32 */
    const float x[3] = {1, 2, 3}, w[3] = {4, 5, 6};
    float y[1];
    op_matmul(x, w, y, 1, 3, 1);
    CHECK_NEAR(y[0], 32.0, TOL);
}

static void add_full_and_broadcast(void) {
    const float a[4] = {1, 2, 3, 4};
    const float full[4] = {1, 1, 2, 2};
    const float row[2] = {10, 20};
    float out[4];
    op_add(a, full, out, 2, 2, 2);
    const float want_full[4] = {2, 3, 5, 6};
    CHECK_ARRAY_NEAR(out, want_full, 4, TOL);
    op_add(a, row, out, 2, 2, 1); /* the row is added to every row */
    const float want_bc[4] = {11, 22, 13, 24};
    CHECK_ARRAY_NEAR(out, want_bc, 4, TOL);
}

static void relu_values(void) {
    const float in[5] = {-2, -0.5f, 0, 0.5f, 3};
    const float want[5] = {0, 0, 0, 0.5f, 3};
    float out[5];
    op_relu(in, out, 5);
    CHECK_ARRAY_NEAR(out, want, 5, 0.0);
}

static void relu_in_place(void) {
    float x[3] = {-1, 2, -3};
    op_relu(x, x, 3);
    CHECK(x[0] == 0 && x[1] == 2 && x[2] == 0);
}

static void sigmoid_values(void) {
    /* sigmoid(0)=0.5, sigmoid(2)=1/(1+e^-2)=0.880797078, sigmoid(-2)=0.119202922 */
    const float in[3] = {0, 2, -2};
    const float want[3] = {0.5f, 0.880797078f, 0.119202922f};
    float out[3];
    op_sigmoid(in, out, 3);
    CHECK_ARRAY_NEAR(out, want, 3, TOL);
}

static void sigmoid_extremes_are_finite(void) {
    const float in[2] = {-1000, 1000};
    float out[2];
    op_sigmoid(in, out, 2);
    CHECK(isfinite(out[0]) && isfinite(out[1]));
    CHECK_NEAR(out[0], 0.0, TOL);
    CHECK_NEAR(out[1], 1.0, TOL);
}

static void softmax_values(void) {
    /* softmax([1,2,3]) = e^[1,2,3] / (e+e^2+e^3) = [0.0900306, 0.2447285, 0.6652410] */
    const float in[3] = {1, 2, 3};
    const float want[3] = {0.0900305732f, 0.244728471f, 0.665240956f};
    float out[3];
    op_softmax(in, out, 1, 3);
    CHECK_ARRAY_NEAR(out, want, 3, TOL);
}

static void softmax_is_stable_and_rowwise(void) {
    /* Row 1 is row 0 shifted by +1000. A naive softmax computes exp(1002) = inf
     * and returns NaN. The stable one gives the same answer as row 0. */
    float x[6] = {1, 2, 3, 1001, 1002, 1003};
    op_softmax(x, x, 2, 3); /* in place */
    const float want[3] = {0.0900305732f, 0.244728471f, 0.665240956f};
    CHECK_ARRAY_NEAR(x, want, 3, TOL);
    CHECK_ARRAY_NEAR(x + 3, want, 3, TOL);
    CHECK_NEAR(x[3] + x[4] + x[5], 1.0, TOL);
}

static void copy_values(void) {
    float out[6];
    op_copy(A23, out, 6);
    CHECK(memcmp(out, A23, sizeof out) == 0);
}

static void linear_hand_computed(void) {
    /* A*B = [[58,64],[139,154]], bias = [1,-200]
     *   linear      = [[59,-136],[140,-46]]
     *   linear_relu = [[59,   0],[140,  0]] */
    const float bias[2] = {1, -200};
    float C[4];
    op_linear(A23, B32, bias, C, 2, 3, 2, 1);
    const float want[4] = {59, -136, 140, -46};
    CHECK_ARRAY_NEAR(C, want, 4, TOL);
    op_linear_relu(A23, B32, bias, C, 2, 3, 2, 1);
    const float want_r[4] = {59, 0, 140, 0};
    CHECK_ARRAY_NEAR(C, want_r, 4, TOL);
}

/* Small deterministic pseudo-random generator (an LCG), so tests are reproducible
 * on every machine. */
static unsigned g_seed = 12345u;
static float frand(void) {
    g_seed = g_seed * 1103515245u + 12345u;
    return (float)((g_seed >> 8) & 0xFFFF) / 32768.0f - 1.0f; /* in [-1, 1) */
}

/* The key fusion guarantee: fused == unfused, bit for bit. That holds for
 * both bias layouts and for awkward (non-power-of-2) sizes. */
static void fused_matches_unfused_bitwise(void) {
    enum { M = 7, K = 13, N = 5 };
    float A[M * K], B[K * N], bias_full[M * N];
    float t1[M * N], t2[M * N], t3[M * N], fused[M * N];
    for (int i = 0; i < M * K; i++) A[i] = frand();
    for (int i = 0; i < K * N; i++) B[i] = frand();
    for (int i = 0; i < M * N; i++) bias_full[i] = frand();

    for (int bias_rows = 1; bias_rows <= M; bias_rows += M - 1) { /* 1, then M */
        /* unfused: MATMUL -> ADD -> RELU through two intermediate buffers */
        op_matmul(A, B, t1, M, K, N);
        op_add(t1, bias_full, t2, M, N, bias_rows);
        op_relu(t2, t3, (size_t)M * N);

        op_linear(A, B, bias_full, fused, M, K, N, bias_rows);
        CHECK(memcmp(fused, t2, sizeof fused) == 0);
        op_linear_relu(A, B, bias_full, fused, M, K, N, bias_rows);
        CHECK(memcmp(fused, t3, sizeof fused) == 0);
    }
}

static void fused_relu_handles_nan_like_relu(void) {
    /* A NaN input gives a NaN pre-activation. op_relu maps it to 0 because
     * (NaN > 0) is false. The fused kernel must do the same. */
    const float A[1] = {NAN}, B[1] = {1}, bias[1] = {0};
    float unf[1], fused[1];
    op_relu(A, unf, 1);
    op_linear_relu(A, B, bias, fused, 1, 1, 1, 1);
    CHECK(unf[0] == 0.0f && fused[0] == 0.0f);
}

void test_ops(void) {
    RUN(matmul_hand_computed);
    RUN(matmul_identity);
    RUN(matmul_vector);
    RUN(add_full_and_broadcast);
    RUN(relu_values);
    RUN(relu_in_place);
    RUN(sigmoid_values);
    RUN(sigmoid_extremes_are_finite);
    RUN(softmax_values);
    RUN(softmax_is_stable_and_rowwise);
    RUN(copy_values);
    RUN(linear_hand_computed);
    RUN(fused_matches_unfused_bitwise);
    RUN(fused_relu_handles_nan_like_relu);
}
