#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_DIR/build"

"$BUILD_DIR/nn-opt" "$SCRIPT_DIR/test_mlp.mlir" > "$BUILD_DIR/test_mlp.ll"
/tank/akash/LLVM-21.1.8-Linux-X64/bin/clang "$BUILD_DIR/test_mlp.ll" "$SCRIPT_DIR/driver.c" -o "$BUILD_DIR/mlp_test"
"$BUILD_DIR/mlp_test"
