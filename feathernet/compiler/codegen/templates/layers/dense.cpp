void layerFunction_Forward(const std::vector<float> &input,
                           std::vector<float> &output,
                           const std::vector<float> &weights,
                           const std::vector<float> &bias) {
  const int input_dim = @INPUT_DIM @;
  const int output_dim = @OUTPUT_DIM @;

  assert(weights.size() == input_dim * output_dim);
  assert(bias.size() == output_dim);

  output.resize(output_dim, 0.0f);

  // Print input data
//  std::cout << "Input data: ";
//  for (const auto& val : input) std::cout << val << " ";
//  std::cout << std::endl;


  // Matrix multiplication
  for (int i = 0; i < output_dim; ++i) {
    for (int j = 0; j < input_dim; ++j) {
      output[i] += input[j] * weights[i * input_dim + j];
    }
    output[i] += bias[i];
  }

  std::cout << "Forward output: ";
  for (const auto& val : output) std::cout << val << " ";
  std::cout << std::endl;
}

void layerFunction_Backward(const std::vector<float> &input,
                            const std::vector<float> &output_grad,
                            std::vector<float> &input_grad,
                            std::vector<float> &weights_grad,
                            std::vector<float> &bias_grad,
                            const std::vector<float> &weights) {
  const int input_dim = @INPUT_DIM @;
  const int output_dim = @OUTPUT_DIM @;

    // Print input data
//  std::cout << "Input data backward: ";
//  for (const auto& val : input) std::cout << val << " ";
//  std::cout << std::endl;

  input_grad.resize(input_dim, 0.0f);
  weights_grad.resize(input_dim * output_dim, 0.0f);
  bias_grad.resize(output_dim, 0.0f);

  for (int i = 0; i < output_dim; ++i) {
    for (int j = 0; j < input_dim; ++j) {
      weights_grad[i * input_dim + j] += output_grad[i] * input[j];
      input_grad[j] += output_grad[i] * weights[i * input_dim + j];
    }
    bias_grad[i] += output_grad[i];
  }

  std::cout << "Backward gradients: ";
  for (const auto& val : input_grad) std::cout << val << " ";
  std::cout << std::endl;
}
