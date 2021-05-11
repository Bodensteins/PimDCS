#ifndef TORCH_LINEAR_H
#define TORCH_LINEAR_H

#pragma once

#include <torch/torch.h>

using namespace torch;
using namespace torch::autograd;
using namespace torch::nn;
using namespace at::native;
namespace F = torch::nn::functional;

class TORCH_API TorchLinearImpl : public Cloneable<TorchLinearImpl> {
public:
  TorchLinearImpl(int64_t in_features, int64_t out_features, Tensor &weight, Tensor &bias)
      : TorchLinearImpl(LinearOptions(in_features, out_features), weight, bias) {}

  explicit TorchLinearImpl(const LinearOptions& options_, Tensor &weight, Tensor &bias)
      : options(options_){
    this->weight = register_parameter("weight",weight);
    if (options.bias()) {
      this->bias = register_parameter("bias", bias);
    } else {
      this->bias = register_parameter("bias", {}, /*requires_grad=*/false);
    }
  }

  void reset() override {
    reset_parameters();
  }

  void reset_parameters() {}

  /// Pretty prints the `Linear` module into the given `stream`.
  void pretty_print(std::ostream &stream) const override {
    stream << std::boolalpha
           << "TorchLinear(in_features=" << options.in_features()
           << ", out_features=" << options.out_features()
           << ", bias=" << options.bias() << ")";
  }

  /// Transforms the `input` tensor by multiplying with the `weight` and
  /// optionally adding the `bias`, if `with_bias` is true in the options.
  Tensor forward(const Tensor &input) {
    return torch::nn::functional::linear(input, this->weight, this->bias);
  }

  /// The options used to configure this module.
  LinearOptions options;

  /// The learned weight.
  Tensor weight;

  /// The learned bias. If `bias` is false in the `options`, this tensor is
  /// undefined.
  Tensor bias;
};

TORCH_MODULE(TorchLinear);

#endif //TORCH_LINEAR_H
