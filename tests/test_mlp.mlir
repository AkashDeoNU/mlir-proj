// Tiny 1-hidden-layer MLP smoke test:
//   %h = relu(x @ W1 + b1)
//   %y = h @ W2 + b2
// Exercises nn.matmul, nn.add (with 1-D bias broadcast), and nn.relu in
// one function so the lowering pipeline gets a workout end-to-end.

func.func @mlp(%x:  tensor<1x4xf64>,
               %W1: tensor<4x3xf64>, %b1: tensor<3xf64>,
               %W2: tensor<3x2xf64>, %b2: tensor<2xf64>) -> tensor<1x2xf64>
    attributes { llvm.emit_c_interface } {
  %z1 = nn.matmul %x,  %W1 : (tensor<1x4xf64>, tensor<4x3xf64>) -> tensor<1x3xf64>
  %a1 = nn.add    %z1, %b1 : (tensor<1x3xf64>, tensor<3xf64>)   -> tensor<1x3xf64>
  %h1 = nn.relu   %a1      : tensor<1x3xf64> -> tensor<1x3xf64>
  %z2 = nn.matmul %h1, %W2 : (tensor<1x3xf64>, tensor<3x2xf64>) -> tensor<1x2xf64>
  %y  = nn.add    %z2, %b2 : (tensor<1x2xf64>, tensor<2xf64>)   -> tensor<1x2xf64>
  return %y : tensor<1x2xf64>
}
