# Research notes (Step 0)

Before writing any code I read the reference projects below for **design ideas only**.
No code was copied from any of them. All TinyNNC code is written from scratch.

## Projects read

| Project | License | What I read | Use in TinyNNC |
|---|---|---|---|
| [shashank976/tinyml-compiler](https://github.com/shashank976/tinyml-compiler) | **None declared** (all rights reserved by default) | README, `include/ir.h`, `src/optimizer.c` | Ideas only, credited |
| [wzh99/relay-mlir](https://github.com/wzh99/relay-mlir) | **None declared** (all rights reserved by default) | README, `lib/Dialect/Relay/Transforms/OpFusion.cpp` | Ideas only, credited |
| [karpathy/llm.c](https://github.com/karpathy/llm.c) | MIT | `train_gpt2.c` (matmul forward, parameter allocation), `test_gpt2.c` (tensor checker) | Ideas only, credited |

Because two of the three repos have no license, copying even small snippets would not be allowed.
Ideas, algorithms and general structure are not covered by copyright, and they are credited in the README.

## What I learned

### 1. tinyml-compiler: the pipeline shape, and the limits of a flat IR
- Pipeline `Model -> IR -> Optimizer -> Codegen -> Simulator` with a plain-text model file, in C,
  with one source file per stage. This is close to what TinyNNC needs.
- Its IR is a **flat array** of nodes. Fusion scans for "`MATMUL` immediately followed by `ADD`" and
  merges them. Redundancy elimination removes `RELU` after `RELU`/`SIGMOID` the same way.
- **Weakness I want to avoid:** adjacency in an array is not the same as data dependence. The pass never
  checks that the `ADD` actually consumes the `MATMUL`'s result, or that nothing else uses that result.
  That works for straight-line MLPs, but it would be wrong on a graph with branches or shared values.

### 2. relay-mlir: how a real compiler decides a fusion is safe
- The fusion matcher starts at a `dense` op and only extends the group to `bias_add` if the dense
  result **has exactly one use**. It then extends to `relu` under the same rule. If the rule fails,
  the group stops early. It does not fail outright.
- Each fused group becomes one "primitive" function, and lowering then turns each group into a single loop nest.
- After lowering, intermediate buffers are allocated and freed **based on their lifetimes**,
  which is the idea behind TinyNNC's liveness-based buffer freeing.
- Shape inference runs as its own pass before optimization.

### 3. llm.c: kernel and testing style in plain C
- Kernels are plain C functions over raw `float*` with explicit sizes. The bias is folded into
  the matmul's output loop (`val = bias[o]; val += ...`). This is the shape a real fused `LINEAR` kernel takes.
- Allocation goes through checked wrappers (`mallocCheck`), so a failed allocation aborts with a
  clear message instead of crashing later.
- Correctness is checked against a reference framework (PyTorch) that writes expected tensors to a file.
  The C side compares element by element with a tolerance and reports the max difference.
- Parameters are put in one large allocation and sliced into pointers, which is cheaper than many small mallocs.

### Background (concepts, not code)
These are the standard designs that the three projects above simplify:
- TVM Relay `FuseOps`: fusion grouped by op pattern (injective, reduction, out-elemwise-fusable).
- ONNX Runtime graph transformers: constant folding, `MatMul+Add -> Gemm` fusion, and pass levels.
- The classic compiler passes (constant folding, DCE, CSE) applied to a dataflow DAG.

## Design choices adopted in TinyNNC

| Decision | Inspired by | Why |
|---|---|---|
| Graph IR is a real DAG: nodes keep input ids, and consumer counts are computed | Weakness in tinyml-compiler, MLIR use-lists | Fusion and DCE need to know who uses each value |
| Fuse only if the intermediate value has exactly one consumer | relay-mlir `hasOneUse()` check | Otherwise the intermediate value is still needed, so it must be kept |
| Two-step fusion: `MATMUL+ADD -> LINEAR`, then `LINEAR+RELU -> LINEAR_RELU` | relay-mlir's dense -> bias_add -> relu chain | Each step is a small, testable rewrite |
| Fused kernels do bias add and ReLU inside the matmul output loop | llm.c `matmul_forward` | Fusion only speeds things up if the intermediate value never goes to memory |
| Free each buffer after its last consumer runs | relay-mlir lifetime-based alloc/dealloc | Lowers peak memory, and the savings can be measured |
| Checked allocation helpers with one clear owner per buffer | llm.c `mallocCheck` | Clean under ASan, and easy to explain |
| Compare against a NumPy reference with a max-abs-diff report | llm.c `test_gpt2.c` | The same idea, using NumPy because PyTorch is not installed on this machine |
| Text model file with one op per line and named values | tinyml-compiler `model.txt`, Relay text form | Easy to write by hand, and gives line-numbered parse errors |

## Toolchain notes for this machine
- Apple M2, macOS (Darwin 25.6), Apple clang 21.0.0, GNU Make 3.81, NumPy 2.2.4, no PyTorch.
- **Valgrind does not run on Apple Silicon.** Memory checking uses ASan + UBSan (`make debug`)
  and macOS's built-in `leaks --atExit` for leak detection. LeakSanitizer is not supported by Apple clang on macOS.
- `libomp` is installed through Homebrew, so the optional OpenMP matmul can be built with `make OMP=1`.
