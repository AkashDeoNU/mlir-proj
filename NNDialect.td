include "mlir/IR/OpBase.td"

def NN_Dialect : Dialect {
    let name = "nn";
    let summary = "a toy neural network dialect to experiment with MLIR";
    let description = "a toy neural network dialect to experiment with MLIR."
                      "it defines only one operation so far - ReLU. nothing else";
    let cppNamespace = "nn";
}
