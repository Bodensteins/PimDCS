//
// Created by 周恒 on 2020/11/26.
//

#ifndef PIMTORCH_PIM_CONV_H
#define PIMTORCH_PIM_CONV_H

#pragma once
#include <torch/torch.h>
#include "logic_array_interface.h"

using namespace torch;
using namespace torch::autograd;
using namespace torch::nn;

class PimConv2DFunction : public Function<PimConv2DFunction> {
public:
  static Tensor forward(
      AutogradContext *ctx, SimpleLogicArray &wb, SimpleLogicArray &wb_t, std::vector<SimpleLogicArray &> prev,
      const Tensor & input, const Tensor & weight, const c10::optional<Tensor>& bias,
      IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups) {
    ctx->save_for_backward({input, weight, bias.has_value() ? bias.value() : Tensor()});
    ctx->saved_data["stride"] = stride;
    ctx->saved_data["padding"] = padding;
    ctx->saved_data["dilation"] = dilation;
    ctx->saved_data["groups"] = groups;



    // ============
    // use intrusive_ptr instead of SimpleLogicArray reference (e.g wb, wb_t, prev)
    c10::intrusive_ptr<SimpleLogicArray> wb_ptr = c10::make_intrusive<SimpleLogicArray>(wb);
    c10::intrusive_ptr<SimpleLogicArray> wb_t_ptr = c10::make_intrusive<SimpleLogicArray>(wb_t);
    c10::intrusive_ptr<SimpleLogicArray> prev_ptr = c10::make_intrusive<SimpleLogicArray>(prev);

    ctx->saved_data["wb"] = wb_ptr;
    ctx->saved_data["wb_t"] = wb_t_ptr;
    ctx->saved_data["prev"] = prev_ptr;

    // write parameters to PIM if is trainable
    if (weight.requires_grad()) {
      prev_ptr->write_mat(input);

      // update parameters, note that weight shape is (out_features, in_features)
      if (bias.defined()) {
        wb_ptr->write_mat(torch::cat({weight.t(), bias.unsqueeze(0)}, 0)); // write transposed weight
      } else {
        wb_ptr->write_mat(torch::cat({weight.t(), torch::zeros({1, weight.size(0)})}, 0));
      }
      wb_t_ptr->write_mat(weight);
    }

    ConstantPad2d m(ConstantPad2dOptions({0, 1, 0, 0}, bias.defined() ? 1 : 0));
    input = m(input);

    Tensor pim_output = wb_ptr->mm(input);   // shape of wb: (in_features, out_features)

    if (!torch::allclose(output, pim_output, 1e-05, 1e-05)) {
      std::cout << "Forward" << std::endl;
      Tensor err = output - pim_output;
      std::cout << err << std::endl;
    }

    return pim_output;
  }

  static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto input = saved[0];
    auto weight = saved[1];
    auto bias = saved[2];

    Tensor grad_output = grad_outputs[0];
    Tensor grad_input = grad_output.mm(weight);
    Tensor grad_weight = grad_output.t().mm(input);
    Tensor grad_bias = Tensor();
    if (bias.defined()) {
      grad_bias = grad_output.sum(0);
    }

    // =============

    // shape of wb_t: (out_features, in_features)
    Tensor pim_grad_input = ctx->saved_data["wb_t"].toCustomClass<SimpleLogicArray>()->mm(grad_output);
    Tensor pim_grad_weight = ctx->saved_data["prev"].toCustomClass<SimpleLogicArray>()->mm(grad_output.t());


    if (!torch::allclose(grad_input, pim_grad_input, 1e-05, 1e-05)) {
      std::cout << "Backward::grad_input" << std::endl;
      Tensor err = grad_input - pim_grad_input;
      std::cout << err << std::endl;
    }

    if (!torch::allclose(grad_weight, pim_grad_weight, 1e-05, 1e-05)) {
      std::cout << "Backward::grad_weight" << std::endl;
      Tensor err = grad_weight - pim_grad_weight;
      std::cout << grad_weight << std::endl;
      std::cout << pim_grad_weight << std::endl;
      std::cout << err << std::endl;
    }

    return {Tensor(), Tensor(), Tensor(), pim_grad_input, pim_grad_weight, grad_bias}; // number of returns should be equal to forward's args.
  }
};


class TORCH_API PimConv2DImpl : public Cloneable<PimConv2DImpl> {
public:
  PimConv2DImpl(int64_t in_features, int64_t out_features, int64_t batch_size)
      : PimConv2DImpl(LinearOptions(in_features, out_features), batch_size) {}
  explicit PimConv2DImpl(const LinearOptions& options_, int64_t batch_size)
      : options(options_), batch_size(batch_size),
        wb(SimpleLogicArray(options_.bias() ? options_.in_features() + 1 : options_.in_features(), options_.out_features())),
        wb_t(SimpleLogicArray(options_.out_features(), options_.in_features())),
        prev(SimpleLogicArray(batch_size, options_.in_features())) {
    reset();
  }

  void reset() override {
    weight = register_parameter("weight",
                                torch::empty({options.out_features(), options.in_features()}));
    if (options.bias()) {
      bias = register_parameter("bias", torch::empty(options.out_features()));
    } else {
      bias = register_parameter("bias", {}, /*requires_grad=*/false);
    }

    reset_parameters();
  }

  void reset_parameters() {
    torch::nn::init::kaiming_uniform_(weight, std::sqrt(5)); // NOLINT(cppcoreguidelines-avoid-magic-numbers)
    if (bias.defined()) {
      int64_t fan_in, fan_out;
      std::tie(fan_in, fan_out) =
          torch::nn::init::_calculate_fan_in_and_fan_out(weight);
      const auto bound = 1 / std::sqrt(fan_in);
      torch::nn::init::uniform_(bias, -bound, bound);
    }
  }

  /// Pretty prints the `Linear` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override {
    stream << std::boolalpha
           << "torch::nn::Linear(in_features=" << options.in_features()
           << ", out_features=" << options.out_features()
           << ", bias=" << options.bias() << ")";
  }

  /// Transforms the `input` tensor by multiplying with the `weight` and
  /// optionally adding the `bias`, if `with_bias` is true in the options.
  Tensor forward(const Tensor& input) {
    return PimLinearFunction::apply(wb, wb_t, prev, input, weight, bias);
  }

  /// The options used to configure this module.
  LinearOptions options;

  /// The learned weight.
  Tensor weight;

  /// The learned bias. If `bias` is false in the `options`, this tensor is
  /// undefined.
  Tensor bias;

  SimpleLogicArray wb;
  SimpleLogicArray wb_t;
  SimpleLogicArray prev;

  int64_t batch_size;
};

TORCH_MODULE(PimConv2D);

#endif //PIMTORCH_PIM_CONV_H
