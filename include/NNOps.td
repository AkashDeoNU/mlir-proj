include "NNDialect.td"

def NN_ReluOp : Op<NN_Dialect, "relu", []> {
    let arguments = (ins F64Tensor:$input);
    let results = (outs F64Tensor:$result);
    let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
    let hasCanonicalizer = 1;
}
