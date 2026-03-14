func.func @double_relu(%arg0: tensor<4xf64>) -> tensor<4xf64> {
  %0 = nn.relu %arg0 : tensor<4xf64> -> tensor<4xf64>
  %1 = nn.relu %0 : tensor<4xf64> -> tensor<4xf64>
  return %1 : tensor<4xf64>
}
