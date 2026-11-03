# TinyNNC build.
#
#   make          build ./tinynnc, the library, and the test binary (release, -O2)
#   make test     run all C tests, then the NumPy cross-check if numpy is installed
#   make crosscheck  only the NumPy cross-check
#   make debug    rebuild with AddressSanitizer + UBSan into build-debug/ and run tests
#   make leaks    run the release tests under macOS `leaks` (Valgrind is unavailable on Apple Silicon)
#   make clean    remove build outputs
#
# Options: OMP=1 enables the OpenMP matmul (added in M5).

CC      ?= cc
STD     := -std=c11
WARN    := -Wall -Wextra -Wpedantic
# OPT and SAN are separate variables so `make debug` can change them without
# losing the warning flags and include path. A CFLAGS given on the command line
# would replace the whole variable.
OPT     ?= -O2
SAN     ?=
CFLAGS  := $(OPT) $(SAN) $(STD) $(WARN) -Iinclude
LDFLAGS := $(SAN)
LDLIBS  := -lm

BUILD   ?= build

# Library sources = every src/*.c except the CLI entry point.
LIB_SRC  := $(filter-out src/main.c,$(wildcard src/*.c))
LIB_OBJ  := $(patsubst src/%.c,$(BUILD)/src/%.o,$(LIB_SRC))
TEST_SRC := $(wildcard tests/*.c)
TEST_OBJ := $(patsubst tests/%.c,$(BUILD)/tests/%.o,$(TEST_SRC))
HEADERS  := $(wildcard include/*.h) $(wildcard tests/*.h)

LIB       := $(BUILD)/libtinynnc.a
TEST_BIN  := $(BUILD)/run_tests
BIN       ?= tinynnc
PYTHON    ?= python3
CROSS_OPTS ?= none,fold,simplify,cse,dce,all

.PHONY: all test unit crosscheck debug leaks bench clean

all: $(LIB) $(TEST_BIN) $(BIN)

$(BIN): $(BUILD)/src/main.o $(LIB)
	$(CC) $(CFLAGS) $(LDFLAGS) $< $(LIB) $(LDLIBS) -o $@

$(BUILD)/src/%.o: src/%.c $(HEADERS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD)/tests/%.o: tests/%.c $(HEADERS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -Itests -c $< -o $@

$(LIB): $(LIB_OBJ)
	@rm -f $@
	ar rcs $@ $^

$(TEST_BIN): $(TEST_OBJ) $(LIB)
	$(CC) $(CFLAGS) $(LDFLAGS) $(TEST_OBJ) $(LIB) $(LDLIBS) -o $@

unit: $(TEST_BIN)
	./$(TEST_BIN)

# The cross-check is skipped (not failed) on machines without NumPy, so that
# `make test` still works on a fresh clone with only a C compiler.
crosscheck: $(BIN)
	$(PYTHON) scripts/crosscheck.py --bin ./$(BIN) --opts $(CROSS_OPTS)

test: unit $(BIN)
	@if $(PYTHON) -c "import numpy" >/dev/null 2>&1; then \
	  $(PYTHON) scripts/crosscheck.py --bin ./$(BIN) --opts $(CROSS_OPTS); \
	else echo "crosscheck: SKIPPED (python3 + numpy not found)"; fi

# Sanitizer build lives in its own directory so it never mixes objects with the release build.
# -fno-sanitize-recover makes any UB report fatal, so `make debug` fails loudly.
debug:
	$(MAKE) BUILD=build-debug BIN=build-debug/tinynnc OPT="-O1 -g -fno-omit-frame-pointer" \
	  SAN="-fsanitize=address,undefined -fno-sanitize-recover=all" test

leaks: $(TEST_BIN) $(BIN)
	leaks --atExit -- ./$(TEST_BIN) | tail -3
	leaks --atExit -- ./$(BIN) examples/redundant.txt --opt all --verify | tail -3

bench:
	@echo "bench: added in milestone M6"

clean:
	rm -rf build build-debug $(BIN)
