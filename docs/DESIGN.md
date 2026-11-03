# TinyNNC design

```
model.txt ──► parser ──► Graph IR ──► pass manager ──► execution plan ──► executor ──► output
              (line-numbered   (DAG +      (fold, simplify,   (topo order +     (kernels, liveness-
               errors, shape   shapes)      cse, dce, fuse)    use counts)       based freeing)
               checks)
```

| File | Responsibility |
|---|---|
| `src/tensor.c` | Tensor struct, checked allocation, overflow-safe shapes, live/peak byte counters |
| `src/ops.c` | Kernels: matmul, add (row broadcast), relu, sigmoid, softmax, copy, fused linear / linear_relu |
| `src/graph.c` | IR: nodes, builders, topological sort, cycle detection, shape inference, printer, `.dot` export, statistics |
| `src/parser.c` | Text model format to Graph |
| `src/passes.c` | Optimization passes and the pass manager |
| `src/executor.c` | Plan creation and execution |
| `src/main.c` | CLI |

## 1. Tensor and memory ownership

`Tensor { float *data; int ndim; int shape[4]; size_t size; }` holds row-major data.

Ownership rules:
- Whoever calls `tensor_new`, `tensor_from` or `tensor_clone` owns the result and must call `tensor_free`. `tensor_free(NULL)` is a no-op.
- Kernels never allocate or free. They only write into buffers the caller provides.
- The **Graph** owns every CONST tensor. The **executor** owns the intermediate buffers it allocates during a run. The **caller** owns the input tensor and the returned output.
- All data allocations go through `tensor.c`, which counts live and peak bytes. That is how peak memory is measured exactly.

## 2. Graph IR

```c
typedef struct Node {
    int id; OpType op; char name[32];
    int ninputs; int inputs[3];   // producer ids (edges stored on the consumer side)
    int ndim; int shape[4];       // output shape, from shape inference
    Tensor *value;                // CONST only
    int dead;                     // removed by a pass
    int line;                     // source line, for error messages
} Node;
```

- **A DAG, not a list.** A value can have any number of consumers. Consumer counts are computed on demand (`graph_count_uses`), so they are never out of date after a rewrite. The graph output counts as one extra consumer.
- **Stable ids.** Passes never delete nodes from the array. They set `dead = 1`. Ids held anywhere stay valid, and a rewrite is just changing fields.
- **Topological order** uses Kahn's algorithm, with consumer lists built in CSR form (O(V+E)). If fewer nodes come out than are live, there is a cycle. The error names a node on it. Edges that point at dead or out-of-range nodes are also reported, so a buggy pass gives an error instead of a crash.
- **Shape inference** checks each op's rule and produces messages like
  `MATMUL 'y': inner dimensions differ: 'x' is [1x4], 'W' is [3x3] (4 != 3)`.

| Op | Inputs | Shape rule |
|---|---|---|
| INPUT / CONST | - | given |
| MATMUL | a (M×K), b (K×N) | M×N |
| ADD | a (M×N), b (M×N or 1×N) | M×N (b's row is broadcast) |
| RELU, SIGMOID, SOFTMAX | x | same as x (softmax is over the last dim) |
| FLATTEN | x (d0, d1, …) | d0 × (d1·d2·…) |
| LINEAR | x (M×K), W (K×N), b (1×N or M×N) | M×N, computes x·W + b |
| LINEAR_RELU | same as LINEAR | M×N, computes max(x·W + b, 0) |

## 3. Model file grammar

```
file       := { statement | comment | blank }
comment    := '#' { any char } end-of-line          (allowed after a statement too)
statement  := input | const | op | output

input      := 'INPUT'  NAME DIM { DIM }             1 to 4 dims, one line, exactly one INPUT
const      := 'CONST'  NAME DIM DIM NUMBER{rows*cols}   values may continue over the following lines
op         := OPNAME   NAME NAME{arity}             one line, operands must already be defined
output     := 'OUTPUT' NAME                         exactly once

OPNAME     := MATMUL | ADD | RELU | SIGMOID | SOFTMAX | FLATTEN | LINEAR | LINEAR_RELU
NAME       := [A-Za-z_][A-Za-z0-9_]{0,30}          unique
DIM        := positive integer, at most 2^24; a tensor has at most 2^24 elements
NUMBER     := anything strtof accepts in full (e.g. 1, -0.5, 3e-2)
```

Because names must be defined before they are used, a model file can never describe a cycle.
Every error is reported as `file:line: error: message`. Shapes are checked as each statement
is parsed, so shape errors carry line numbers too. The parser bounds every size before
allocating, rejects NUL bytes, and is fuzzed with 3000 random mutations under
ASan/UBSan (`tests/test_parser.c`).

## 4. Optimization passes

Every pass rewrites the graph in place, returns how many nodes it changed, and must preserve
the output for every input. Below is each pass's **safety condition**: the reason the rewrite
cannot change the result.

| Pass | What it does | Why it is safe |
|---|---|---|
| **Constant folding** | A node whose inputs are all CONST is evaluated at compile time and becomes a CONST. Runs in topological order, so whole constant chains fold in one sweep. | The value does not depend on the run-time input. It uses the same kernel (`exec_node`) as the executor, so the folded value is bit-identical. |
| **Algebraic simplification** | `RELU(RELU(v))→RELU(v)`, `RELU(LINEAR_RELU)→LINEAR_RELU`, `v+0→v`, `0+v→v`, `v·I→v`, `I·v→v`, `FLATTEN(v)→v` when v is already that 2-D shape. | ReLU is idempotent. Adding ±0 and multiplying by I leave values unchanged. **Not done:** `RELU(SIGMOID(v))→SIGMOID(v)`, because RELU(NaN)=0 but SIGMOID(NaN)=NaN. **Assumption:** `v·I→v` assumes finite activations (inf·0 = NaN). |
| **CSE** | Two nodes with the same op, the same input ids (in the same order) and the same shape become one. CONSTs with identical bits are merged. Cascades in topological order. | Every op is a pure function. ADD(a,b) and ADD(b,a) are not merged, to keep the rule simple and exact. |
| **DCE** | Mark-and-sweep: mark everything reachable backwards from the output, and kill the rest (except INPUT). | Ops have no side effects, so an unreachable node cannot affect the output. |
| **Fusion: LINEAR** | `ADD(MATMUL(x,W), b)` or `ADD(b, MATMUL(x,W))` → `LINEAR(x,W,b)` | **Fuse only if the MATMUL result has exactly one consumer and is not the graph output.** Also requires the MATMUL result to have the full output shape and b to be 1×N or M×N. Float `+` is commutative in IEEE 754, so both operand orders are exact. |
| **Fusion: LINEAR_RELU** | `RELU(LINEAR(...))` → `LINEAR_RELU(...)` | Same single-consumer rule for the LINEAR value. |

**Why fusion needs the single-consumer rule.** A fused kernel never writes the intermediate
value (for example, the MATMUL result) to memory. If another node also reads that value, it would
have nothing to read. See `examples/shared_value.txt`: there, `m = MATMUL(x, W)` feeds both
`ADD` and `SIGMOID`, so the pass leaves it unfused. Five tests check cases where fusion must be refused.

**Pass manager.** The cleanup passes (fold → simplify → CSE → DCE) repeat until a round
changes nothing (a fixed point, capped at 16 rounds), because each one can create work for
another. Fusion runs once at the end, on the cleaned-up graph. Ordering matters. In
`examples/redundant.txt` the second MATMUL has a duplicate that feeds a dead debug branch.
Only after CSE and DCE does it have one consumer, and only then can it fuse. Afterwards, shape
inference runs again to check that the graph is still valid. With a log stream, the manager
prints per-pass counts and a before/after table (ops, FLOPs, activation bytes, constant bytes).

## 5. Fused kernels

`op_linear` and `op_linear_relu` compute one output row of the matmul. While that row is
still in L1 cache they add the bias and apply ReLU, all inside a single loop over rows.
Unfused, the full M×N result is written, read back by ADD, written again, and read back again
by RELU, and two extra intermediate buffers are needed.

The matmul uses **i-k-j** loop order, so the inner loop walks rows of B and C contiguously
(cache-friendly and auto-vectorizable).

The fused and unfused paths share the same row routine. The dot product is accumulated in the
same order, and the bias is added afterwards in both. So **fused output is bit-identical to
unfused**, and the tests check this with `memcmp`.

## 6. Executor and memory model

- `plan_create` (once per graph) computes the topological order and each node's use count.
- `plan_run` walks the order. It allocates each node's output, runs the kernel, and then
  decrements the remaining-use count of each input. **When a count reaches 0, that buffer is
  freed immediately** (liveness-based freeing). A value read twice, as in `ADD(x, x)`, is freed
  only after both reads. A node with no consumers is freed right after it runs.
- The input and CONST tensors are borrowed and never freed by the executor. The output buffer
  is handed to the caller.
- Peak activation memory is measured, not estimated: the tensor module's peak counter is reset
  at the start of a run and read at the end.
- `--no-reuse` keeps every intermediate alive until the end, which gives the baseline for the
  memory savings.

## 7. Testing strategy

| Layer | Tests |
|---|---|
| Kernels | hand-computed values; numerical stability of softmax and sigmoid; fused == unfused bit for bit |
| Graph | topological order, multi-consumer use counts, cycles, self-loops, dangling edges, shape errors |
| Parser | 23 malformed-input cases with exact line numbers and messages; a 3000-iteration mutation fuzz test |
| Executor | hand-computed MLP; liveness peak memory (16000 → 8000 bytes); double-read freeing; no leaks |
| Passes | unit tests per pass, including must-not-apply cases; 300 random DAGs × 7 pass sets compared to unoptimized |
| End to end | `scripts/crosscheck.py`: 8 generated models × 7 `--opt` settings vs an independent NumPy implementation |
| Memory safety | everything above under ASan + UBSan (`make debug`) and macOS `leaks` (`make leaks`) |
