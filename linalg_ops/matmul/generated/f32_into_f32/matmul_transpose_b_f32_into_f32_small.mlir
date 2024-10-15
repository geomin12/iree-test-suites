func.func @matmul_accumulate_DYNxDYNxf32_times_DYNxDYNxf32_into_DYNxDYNxf32(%lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>, %acc: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<?x?xf32>, tensor<?x?xf32>) outs(%acc: tensor<?x?xf32>) -> tensor<?x?xf32>
  return %result: tensor<?x?xf32>
}

func.func @matmul_accumulate_1x1xf32_times_1x1xf32_into_1x1xf32(%lhs: tensor<1x1xf32>, %rhs: tensor<1x1xf32>, %acc: tensor<1x1xf32>) -> tensor<1x1xf32> {
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<1x1xf32>, tensor<1x1xf32>) outs(%acc: tensor<1x1xf32>) -> tensor<1x1xf32>
  return %result: tensor<1x1xf32>
}

func.func @matmul_DYNxDYNxf32_times_DYNxDYNxf32_into_DYNxDYNxf32(%lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %acc_dim0 = tensor.dim %lhs, %c0 : tensor<?x?xf32>
  %acc_dim1 = tensor.dim %rhs, %c1 : tensor<?x?xf32>
  %init_acc = tensor.empty(%acc_dim0, %acc_dim1) : tensor<?x?xf32>
  %c0_acc_type = arith.constant 0.0: f32
  %acc = linalg.fill ins(%c0_acc_type : f32) outs(%init_acc : tensor<?x?xf32>) -> tensor<?x?xf32>
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<?x?xf32>, tensor<?x?xf32>) outs(%acc: tensor<?x?xf32>) -> tensor<?x?xf32>
  return %result: tensor<?x?xf32>
}

func.func @matmul_1x1xf32_times_1x1xf32_into_1x1xf32(%lhs: tensor<1x1xf32>, %rhs: tensor<1x1xf32>) -> tensor<1x1xf32> {
  %init_acc = tensor.empty() : tensor<1x1xf32>
  %c0_acc_type = arith.constant 0.0: f32
  %acc = linalg.fill ins(%c0_acc_type : f32) outs(%init_acc : tensor<1x1xf32>) -> tensor<1x1xf32>
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<1x1xf32>, tensor<1x1xf32>) outs(%acc: tensor<1x1xf32>) -> tensor<1x1xf32>
  return %result: tensor<1x1xf32>
}

func.func @matmul_accumulate_2x2xf32_times_2x2xf32_into_2x2xf32(%lhs: tensor<2x2xf32>, %rhs: tensor<2x2xf32>, %acc: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<2x2xf32>, tensor<2x2xf32>) outs(%acc: tensor<2x2xf32>) -> tensor<2x2xf32>
  return %result: tensor<2x2xf32>
}

func.func @matmul_accumulate_4x4xf32_times_4x4xf32_into_4x4xf32(%lhs: tensor<4x4xf32>, %rhs: tensor<4x4xf32>, %acc: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<4x4xf32>, tensor<4x4xf32>) outs(%acc: tensor<4x4xf32>) -> tensor<4x4xf32>
  return %result: tensor<4x4xf32>
}

func.func @matmul_accumulate_8x8xf32_times_8x8xf32_into_8x8xf32(%lhs: tensor<8x8xf32>, %rhs: tensor<8x8xf32>, %acc: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<8x8xf32>, tensor<8x8xf32>) outs(%acc: tensor<8x8xf32>) -> tensor<8x8xf32>
  return %result: tensor<8x8xf32>
}

func.func @matmul_accumulate_9x9xf32_times_9x9xf32_into_9x9xf32(%lhs: tensor<9x9xf32>, %rhs: tensor<9x9xf32>, %acc: tensor<9x9xf32>) -> tensor<9x9xf32> {
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<9x9xf32>, tensor<9x9xf32>) outs(%acc: tensor<9x9xf32>) -> tensor<9x9xf32>
  return %result: tensor<9x9xf32>
}

func.func @matmul_accumulate_6x13xf32_times_3x13xf32_into_6x3xf32(%lhs: tensor<6x13xf32>, %rhs: tensor<3x13xf32>, %acc: tensor<6x3xf32>) -> tensor<6x3xf32> {
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<6x13xf32>, tensor<3x13xf32>) outs(%acc: tensor<6x3xf32>) -> tensor<6x3xf32>
  return %result: tensor<6x3xf32>
}

func.func @matmul_15x37xf32_times_7x37xf32_into_15x7xf32(%lhs: tensor<15x37xf32>, %rhs: tensor<7x37xf32>) -> tensor<15x7xf32> {
  %init_acc = tensor.empty() : tensor<15x7xf32>
  %c0_acc_type = arith.constant 0.0: f32
  %acc = linalg.fill ins(%c0_acc_type : f32) outs(%init_acc : tensor<15x7xf32>) -> tensor<15x7xf32>
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<15x37xf32>, tensor<7x37xf32>) outs(%acc: tensor<15x7xf32>) -> tensor<15x7xf32>
  return %result: tensor<15x7xf32>
}

func.func @matmul_accumulate_81x19xf32_times_41x19xf32_into_81x41xf32(%lhs: tensor<81x19xf32>, %rhs: tensor<41x19xf32>, %acc: tensor<81x41xf32>) -> tensor<81x41xf32> {
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<81x19xf32>, tensor<41x19xf32>) outs(%acc: tensor<81x41xf32>) -> tensor<81x41xf32>
  return %result: tensor<81x41xf32>
}

func.func @matmul_accumulate_1x10xf32_times_10x10xf32_into_1x10xf32(%lhs: tensor<1x10xf32>, %rhs: tensor<10x10xf32>, %acc: tensor<1x10xf32>) -> tensor<1x10xf32> {
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<1x10xf32>, tensor<10x10xf32>) outs(%acc: tensor<1x10xf32>) -> tensor<1x10xf32>
  return %result: tensor<1x10xf32>
}

func.func @matmul_1x10xf32_times_10x10xf32_into_1x10xf32(%lhs: tensor<1x10xf32>, %rhs: tensor<10x10xf32>) -> tensor<1x10xf32> {
  %init_acc = tensor.empty() : tensor<1x10xf32>
  %c0_acc_type = arith.constant 0.0: f32
  %acc = linalg.fill ins(%c0_acc_type : f32) outs(%init_acc : tensor<1x10xf32>) -> tensor<1x10xf32>
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<1x10xf32>, tensor<10x10xf32>) outs(%acc: tensor<1x10xf32>) -> tensor<1x10xf32>
  return %result: tensor<1x10xf32>
}

func.func @matmul_accumulate_10x1xf32_times_10x1xf32_into_10x10xf32(%lhs: tensor<10x1xf32>, %rhs: tensor<10x1xf32>, %acc: tensor<10x10xf32>) -> tensor<10x10xf32> {
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<10x1xf32>, tensor<10x1xf32>) outs(%acc: tensor<10x10xf32>) -> tensor<10x10xf32>
  return %result: tensor<10x10xf32>
}

func.func @matmul_accumulate_10x10xf32_times_1x10xf32_into_10x1xf32(%lhs: tensor<10x10xf32>, %rhs: tensor<1x10xf32>, %acc: tensor<10x1xf32>) -> tensor<10x1xf32> {
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<10x10xf32>, tensor<1x10xf32>) outs(%acc: tensor<10x1xf32>) -> tensor<10x1xf32>
  return %result: tensor<10x1xf32>
}

func.func @matmul_10x10xf32_times_1x10xf32_into_10x1xf32(%lhs: tensor<10x10xf32>, %rhs: tensor<1x10xf32>) -> tensor<10x1xf32> {
  %init_acc = tensor.empty() : tensor<10x1xf32>
  %c0_acc_type = arith.constant 0.0: f32
  %acc = linalg.fill ins(%c0_acc_type : f32) outs(%init_acc : tensor<10x1xf32>) -> tensor<10x1xf32>
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<10x10xf32>, tensor<1x10xf32>) outs(%acc: tensor<10x1xf32>) -> tensor<10x1xf32>
  return %result: tensor<10x1xf32>
}

