#ifndef TORCH_CONV_H
#define TORCH_CONV_H

#pragma once

#include <torch/torch.h>

using namespace torch;
using namespace torch::autograd;
using namespace torch::nn;
using namespace at::native;
namespace F = torch::nn::functional;

class TORCH_API TorchConv2dImpl : public Cloneable<TorchConv2dImpl> {
public:
  TorchConv2dImpl(int64_t input_channels,
                  int64_t output_channels,
                  ExpandingArray<2> kernel_size,
                  Tensor &weight,
                  Tensor &bias)
      : TorchConv2dImpl(Conv2dOptions(input_channels, output_channels, kernel_size), weight, bias) {}

  explicit TorchConv2dImpl(Conv2dOptions options_, Tensor &weight, Tensor &bias)
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
    stream << "PIM::PimConv2d"
           << "(" << options.in_channels()
           << ", " << options.out_channels()
           << ", kernel_size=" << options.kernel_size()
           << ", stride=" << options.stride();
    if (*options.padding() != *ExpandingArray<2>(0)) {
      stream << ", padding=" << options.padding();
    }
    stream << ", dilation=" << options.dilation();
    stream << ", output_padding=0";
    stream << ", groups=" << options.groups();

    if (!options.bias()) {
      stream << ", bias=" << std::boolalpha << false;
    }
    if (!c10::get_if<enumtype::kZeros>(&options.padding_mode())) {
      stream << ", padding_mode=" << enumtype::get_enum_name(options.padding_mode());
    }
    stream << ")" << std::endl;
  }

  /// Transforms the `input` tensor by multiplying with the `weight` and
  /// optionally adding the `bias`, if `with_bias` is true in the options.
  Tensor forward(const Tensor &input) {
    return torch::nn::functional::conv2d(input, this->weight,
              torch::nn::functional::Conv2dFuncOptions()
              .bias(this->bias)
              .stride(options.stride())
              .padding(options.padding())
              .dilation(options.dilation()));
  }

  /// The options used to configure this module.
  Conv2dOptions options;

  /// The learned weight.
  Tensor weight;

  /// The learned bias. If `bias` is false in the `options`, this tensor is
  /// undefined.
  Tensor bias;
};

TORCH_MODULE(TorchConv2d);

#endif //TORCH_CONV_H
