include "NNDialect.td"

def NN_ReluOp : Op<NN_Dialect, "relu", []> {
    let arguments = (ins F64Tensor:$input);
    let results = (outs F64Tensor:$result);
    let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
    let hasCanonicalizer = 1;
}

// 2-D matrix multiply: %lhs : tensor<MxKxf64>, %rhs : tensor<KxNxf64>
// produces a tensor<MxNxf64>. The shape constraint is checked at lowering
// time, not by the verifier.
def NN_MatmulOp : Op<NN_Dialect, "matmul", []> {
    let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
    let results = (outs F64Tensor:$result);
    let assemblyFormat =
        "$lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` "
        "`->` type($result)";
}

// Elementwise add with right-side broadcasting. If lhs and rhs have the
// same rank, this is a plain elementwise add. If rhs has lower rank, it
// is broadcast over the leading dims of lhs (the bias-add pattern,
// e.g. tensor<BxNxf64> + tensor<Nxf64> -> tensor<BxNxf64>).
def NN_AddOp : Op<NN_Dialect, "add", []> {
    let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
    let results = (outs F64Tensor:$result);
    let assemblyFormat =
        "$lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` "
        "`->` type($result)";
}
