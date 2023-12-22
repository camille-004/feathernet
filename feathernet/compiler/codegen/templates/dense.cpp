#include <cassert>
#include <iostream>
#include <vector>

void denseLayer(const std::vector<float> &input, std::vector<float> &output,
                const std::vector<float> &weights,
                const std::vector<float> &bias) {
  const int input_dim = @INPUT_DIM @;
  const int output_dim = @OUTPUT_DIM @;

  assert(weights.size() == input_dim * output_dim);
  assert(bias.size() == output_dim);

  output.resize(output_dim, 0.0f);

  // Matrix multiplication
  for (int i = 0; i < output_dim; ++i) {
    for (int j = 0; j < input_dim; ++j) {
      output[i] += input[j] * weights[i * input_dim + j];
    }
    output[i] += bias[i];
  }
}

int main() {
  const int input_dim = @INPUT_DIM @;
  const int output_dim = @OUTPUT_DIM @;

  std::vector<float> input(input_dim, 1.0f);

  std::vector<float> weights = @WEIGHTS @;
  std::vector<float> bias = @BIASES @;

  std::vector<float> output(output_dim);

  denseLayer(input, output, weights, bias);

  for (int i = 0; i < output_dim; ++i) {
    std::cout << output[i] << " ";
  }
  std::cout << std::endl;

  return 0;
}
