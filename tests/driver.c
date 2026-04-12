// Driver for the lowered nn `mlp` kernel.
//
// Compiles test_mlp.mlir via nn-opt, then this file provides main(),
// populates weights, calls the compiled MLP, and prints the result.
//
// The MLIR function is declared with `attributes { llvm.emit_c_interface }`,
// so func-to-llvm emits a C-callable wrapper `_mlir_ciface_mlp` that
// takes all MemRef descriptors by pointer.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// ---------------------------------------------------------------------------
// MemRef descriptor types matching the MLIR ABI.
// ---------------------------------------------------------------------------

typedef struct {
  double *allocated;
  double *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} MemRef1D_f64;

typedef struct {
  double *allocated;
  double *aligned;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
} MemRef2D_f64;

// ---------------------------------------------------------------------------
// Helpers to build descriptors from stack / heap arrays.
// ---------------------------------------------------------------------------

static MemRef1D_f64 make_1d(double *data, int64_t n) {
  return (MemRef1D_f64){
      .allocated = data,
      .aligned = data,
      .offset = 0,
      .sizes = {n},
      .strides = {1},
  };
}

static MemRef2D_f64 make_2d(double *data, int64_t rows, int64_t cols) {
  return (MemRef2D_f64){
      .allocated = data,
      .aligned = data,
      .offset = 0,
      .sizes = {rows, cols},
      .strides = {cols, 1}, // row-major
  };
}

// ---------------------------------------------------------------------------
// The C-interface wrapper emitted by func-to-llvm.
//
// func.func @mlp(%x:  tensor<1x4xf64>,
//                %W1: tensor<4x3xf64>, %b1: tensor<3xf64>,
//                %W2: tensor<3x2xf64>, %b2: tensor<2xf64>)
//     -> tensor<1x2xf64>
//
// With llvm.emit_c_interface the result (tensor<1x2xf64>) becomes the
// first pointer argument; the five inputs follow.
// ---------------------------------------------------------------------------

extern void _mlir_ciface_mlp(MemRef2D_f64 *result, MemRef2D_f64 *x,
                              MemRef2D_f64 *W1, MemRef2D_f64 *b1,
                              MemRef2D_f64 *W2, MemRef2D_f64 *b2);

int main(void) {
  // Input: 1 sample, 4 features.
  double x_data[1 * 4] = {1.0, 2.0, 3.0, 4.0};
  MemRef2D_f64 x = make_2d(x_data, 1, 4);

  // Hidden layer: 4 -> 3.
  //   W1[i][j] = 0.1*(3*i + j + 1)
  double W1_data[4 * 3] = {
      0.1, 0.2, 0.3, //
      0.4, 0.5, 0.6, //
      0.7, 0.8, 0.9, //
      1.0, 1.1, 1.2, //
  };
  double b1_data[1 * 3] = {0.1, 0.2, 0.3};
  MemRef2D_f64 W1 = make_2d(W1_data, 4, 3);
  MemRef2D_f64 b1 = make_2d(b1_data, 1, 3);

  // Output layer: 3 -> 2.
  double W2_data[3 * 2] = {
      0.1, 0.2, //
      0.3, 0.4, //
      0.5, 0.6, //
  };
  double b2_data[1 * 2] = {0.1, 0.2};
  MemRef2D_f64 W2 = make_2d(W2_data, 3, 2);
  MemRef2D_f64 b2 = make_2d(b2_data, 1, 2);

  // Call the compiled MLP.
  MemRef2D_f64 result = {0};
  _mlir_ciface_mlp(&result, &x, &W1, &b1, &W2, &b2);

  // Print the 1x2 output.
  //
  // Expected (hand-computed):
  //   z1 = x @ W1     = [7.0, 8.0, 9.0]
  //   a1 = z1 + b1    = [7.1, 8.2, 9.3]
  //   h1 = relu(a1)   = [7.1, 8.2, 9.3]    (all positive)
  //   z2 = h1 @ W2    = [7.82, 10.28]
  //   y  = z2 + b2    = [7.92, 10.48]
  printf("mlp output (1x2):\n");
  for (int64_t r = 0; r < result.sizes[0]; ++r) {
    for (int64_t c = 0; c < result.sizes[1]; ++c) {
      int64_t idx = result.offset + r * result.strides[0] + c * result.strides[1];
      printf("  [%lld][%lld] = %f\n", (long long)r, (long long)c,
             result.aligned[idx]);
    }
  }

  free(result.allocated);
  return 0;
}
