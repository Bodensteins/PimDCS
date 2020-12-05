//
// Created by 周恒 on 2020/11/22.
//

#ifndef PIMTORCH_PIM_LINEAR_H
#define PIMTORCH_PIM_LINEAR_H

#pragma once
#include <torch/torch.h>
#include "logic_array_interface.h"

using namespace torch;
using namespace torch::autograd;
using namespace torch::nn;

namespace PIM {
  template<typename PimType>
  class PimLinearFunction : public Function<PimLinearFunction<PimType>> {

  public:
    // Note that both forward and backward are static functions

    // bias is an optional argument
    static Tensor forward(
        AutogradContext *ctx, PimType &wb, PimType &wb_t, PimType &prev,
        Tensor input, Tensor weight, Tensor bias = Tensor()) {
      ctx->save_for_backward({input, weight, bias});
      Tensor output = input.mm(weight.t());
      if (bias.defined()) {
        output += bias.unsqueeze(0).expand_as(output);
      }

      // ============
      // use intrusive_ptr instead of SimpleLogicArray reference (e.g wb_ptr, wb_t_ptr, prev_ptr)
      c10::intrusive_ptr<PimType> wb_ptr = c10::make_intrusive<PimType>(wb);
      c10::intrusive_ptr<PimType> wb_t_ptr = c10::make_intrusive<PimType>(wb_t);
      c10::intrusive_ptr<PimType> prev_ptr = c10::make_intrusive<PimType>(prev);

      ctx->saved_data["wb_ptr"] = wb_ptr;
      ctx->saved_data["wb_t_ptr"] = wb_t_ptr;
      ctx->saved_data["prev_ptr"] = prev_ptr;

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

      Tensor pim_output = wb_ptr->mm(input);   // shape of wb_ptr: (in_features, out_features)

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

      // shape of wb_t_ptr: (out_features, in_features)
      Tensor pim_grad_input = ctx->saved_data["wb_t_ptr"].toCustomClass<PimType>()->mm(grad_output);
      Tensor pim_grad_weight = ctx->saved_data["prev_ptr"].toCustomClass<PimType>()->mm(grad_output.t());


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

      // number of returns should be equal to forward's args.
      return {Tensor(), Tensor(), Tensor(), pim_grad_input, pim_grad_weight, grad_bias};
    }
  };


  class TORCH_API PimLinearImpl : public Cloneable<PimLinearImpl> {
  public:
    PimLinearImpl(int64_t in_features, int64_t out_features, int64_t batch_size, PimArrayType pim_type)
        : PimLinearImpl(LinearOptions(in_features, out_features), batch_size, pim_type) {}

    explicit PimLinearImpl(const LinearOptions &options_, int64_t batch_size, PimArrayType pim_type)
        : options(options_), batch_size(batch_size), pim_type(pim_type) {
      create_pim_array(wb_ptr, {
          options_.bias() ? options_.in_features() + 1 : options_.in_features(), options_.out_features()
        }, pim_type);
      create_pim_array(wb_t_ptr, {options_.out_features(), options_.in_features()}, pim_type);
      create_pim_array(prev_ptr, {batch_size, options_.in_features()}, pim_type);
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
    void pretty_print(std::ostream &stream) const override {
      stream << std::boolalpha
             << "PIM::PimLinear(in_features=" << options.in_features()
             << ", out_features=" << options.out_features()
             << ", bias=" << options.bias() << ")";
    }

    /// Transforms the `input` tensor by multiplying with the `weight` and
    /// optionally adding the `bias`, if `with_bias` is true in the options.
    Tensor forward(const Tensor &input) {
      switch (pim_type) {
        case PimArrayType::simple_logic_array:
          return PimLinearFunction<SimpleLogicArray>::apply(
              *dynamic_cast<SimpleLogicArray*>(wb_ptr.get()),
              *dynamic_cast<SimpleLogicArray*>(wb_t_ptr.get()),
              *dynamic_cast<SimpleLogicArray*>(prev_ptr.get()),
              input, weight, bias);
        case PimArrayType::wb_logic_array:
          C10_THROW_ERROR(Error, "This PIM type is not implemented!");
      }
    }

    /// The options used to configure this module.
    LinearOptions options;

    /// The learned weight.
    Tensor weight;

    /// The learned bias. If `bias` is false in the `options`, this tensor is
    /// undefined.
    Tensor bias;

    PimPtr wb_ptr;
    PimPtr wb_t_ptr;
    PimPtr prev_ptr;
    PimArrayType pim_type;
    int64_t batch_size;
  };

/// A `ModuleHolder` subclass for `LinearImpl`.
/// See the documentation for `LinearImpl` class to learn what methods it
/// provides, and examples of how to use `Linear` with `torch::nn::LinearOptions`.
/// See the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
  TORCH_MODULE(PimLinear);
}
#endif //PIMTORCH_PIM_LINEAR_H
